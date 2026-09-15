"""Atomic opt-in model/index bundle; no environment files or aliases are changed."""
from __future__ import annotations
import json
from pathlib import Path

def apply_profile(settings):
    raw=getattr(settings,'RETRIEVAL_PROFILE_PATH','')
    if not raw:return settings
    path=Path(raw).expanduser()
    if not path.is_absolute():raise ValueError('RETRIEVAL_PROFILE_PATH must be absolute')
    profile=json.loads(path.read_text(encoding='utf-8'))
    if profile.get('schema_version')!=1:raise ValueError('Unsupported retrieval profile')
    if profile.get('approval')!='validated':raise ValueError('Retrieval profile has not passed validation')
    overrides={}
    for role,model_key,revision_key,index_key,dim_key in [
        ('text','QWEN_TEXT_EMBEDDING_MODEL_ID','QWEN_TEXT_EMBEDDING_MODEL_REVISION','OPENSEARCH_INDEX_ARTIFACT','ARTIFACT_TEXT_EMBEDDING_DIMENSION'),
        ('vl','QWEN_MULTIMODAL_EMBEDDING_MODEL_ID','QWEN_MULTIMODAL_EMBEDDING_MODEL_REVISION','OPENSEARCH_INDEX_IMAGE','IMAGE_MULTIMODAL_EMBEDDING_DIMENSION')]:
        item=profile[role]
        if not item.get('model') or not item.get('revision') or not item.get('index') or int(item.get('dimension',0))<1:
            raise ValueError('Incomplete model/index bundle')
        if item.get('normalization')!='l2':raise ValueError('Only normalized embedding spaces are supported')
        overrides.update({model_key:item.get('local_path') or item['model'],revision_key:item['revision'],index_key:item['index'],dim_key:item['dimension']})
    # Clear legacy model aliases atomically with the new identifiers.
    overrides.update(TEXT_EMBEDDING_MODEL=None,MULTIMODAL_EMBEDDING_MODEL=None,USE_OPENROUTER_BGE_M3=False)
    for role,prefix in [('text_reranker','RERANKER'),('vl_reranker','VL_RERANKER')]:
        rr=profile.get(role)
        overrides['CHAT_ENABLE_RERANKING' if role=='text_reranker' else 'CHAT_ENABLE_VL_RERANKING']=bool(rr)
        if rr:
            if not rr.get('model') or not rr.get('revision'):raise ValueError('Unpinned reranker')
            overrides[prefix+'_MODEL_ID']=rr['model'];overrides[prefix+'_MODEL_REVISION']=rr['revision']
            overrides[prefix+'_CANDIDATES']=int(rr['candidates'])
    # Revalidate through pydantic; model_copy(update=...) would bypass validation.
    return type(settings)(_env_file=None,**{**settings.model_dump(),**overrides})

def verify_profile_indices(settings,client):
    raw=getattr(settings,'RETRIEVAL_PROFILE_PATH','')
    if not raw:return
    profile=json.loads(Path(raw).read_text(encoding='utf-8'))
    for role,field in [('text','text_embedding'),('vl','visual_embedding')]:
        item=profile[role];name=item['index']
        mapping=client.indices.get_mapping(index=name)[name]['mappings']
        space=mapping.get('_meta',{}).get('space',{})
        if mapping['properties'][field]['dimension']!=item['dimension']:
            raise ValueError('Index dimension differs from retrieval profile')
        for key,expected in [('model_id',item['model']),('revision',item['revision']),('normalization','l2')]:
            if space.get(key)!=expected:raise ValueError('Index embedding space differs from retrieval profile: '+key)
        if mapping.get('_meta',{}).get('snapshot_id')!=profile.get('snapshot_id'):
            raise ValueError('Index snapshot differs from retrieval profile')
