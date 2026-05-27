# OpenSearch pagination methodology

## Problem

The previous retrieval pagination calculated vector `k` from the requested page:

```text
k = (from + size) * 3
```

This made later pages search a larger vector candidate set than earlier pages. With a boosted clause such as `in_tour`, a document that was not considered on page 1 could enter the candidate set on page 2 and move up in the score order. That makes pagination unstable because pages are not slices of the same ranked result set.

## Implemented approach

The backend now uses a fixed retrieval window per search:

```text
text retrieval window  = CHAT_RETRIEVAL_PAGINATION_WINDOW
image retrieval window = CHAT_IMAGE_RETRIEVAL_PAGINATION_WINDOW
```

Both default to `150`.

For a given search, the OpenSearch query keeps the same candidate universe across pages:

```text
page 1: from=0,  size=15, k=150
page 2: from=15, size=15, k=150
page 3: from=30, size=15, k=150
```

For hybrid text search, the same fixed value is also used as `pagination_depth`, so OpenSearch hybrid normalization receives a stable depth instead of a depth derived from the current page.

## How `in_tour` behaves

`in_tour` is still an OpenSearch score boost:

```json
{
  "term": {
    "in_tour": {
      "value": true,
      "boost": 5.0
    }
  }
}
```

The boost is applied inside each OpenSearch query before `from`/`size` trims the page. With the fixed window, each page is now trimmed from the same vector/hybrid candidate window, so `in_tour` no longer benefits from a larger candidate set on later pages.

## Result limits

The API reports `results_total` capped to the retrieval window. For example, with a window of `150`, the UI should not advertise page results beyond those 150 stable candidates.

This is intentional: approximate nearest-neighbor retrieval is not a good basis for unbounded page-number pagination. A bounded, stable result window is preferable for this assistant because users inspect a small number of museum objects rather than thousands of vector-nearest candidates.

## Why not PIT + search_after yet

OpenSearch recommends Point in Time (PIT) with `search_after` for deep, consistent pagination:

- https://docs.opensearch.org/3.2/search-plugins/searching-data/paginate/
- https://docs.opensearch.org/3.0/search-plugins/searching-data/point-in-time/

That is the right design when the API is cursor-based. This application currently exposes page-number pagination, so switching to PIT + `search_after` would require changing the frontend/backend contract to pass a cursor containing the last hit sort values and PIT ID.

For vector search, PIT + `search_after` still needs a sufficiently large fixed `k`, because the k-NN query only exposes the nearest-neighbor window selected by `k`.

## Recommended future cursor design

If the frontend moves from page numbers to cursors, use:

```json
{
  "size": 15,
  "pit": {
    "id": "<pit_id>",
    "keep_alive": "2m"
  },
  "query": {
    "...": "same query as page 1"
  },
  "sort": [
    { "_score": "desc" },
    { "artifact_id": "asc" }
  ],
  "search_after": ["<last_score>", "<last_artifact_id>"]
}
```

For image search, use a stable tiebreaker such as `image_id`. For text/artifact search, use `artifact_id` or another stable keyword field. The first request omits `search_after`; every following request uses the `sort` array returned by the last hit of the previous page.

## Changed files

- `app/core/config.py`: added fixed pagination window settings.
- `app/services/opensearch_client.py`: added fixed `retrieval_window_size` support for text, image, and multiview retrieval.
- `app/services/chat_service.py`: stores the window in the session retrieval request and caps reported totals.
- `app/services/model_retrieval.py`: uses the fixed image retrieval window for model multiview search.
- `tests/test_opensearch_mapping_fields.py`: verifies fixed `k` and `pagination_depth` behavior.
