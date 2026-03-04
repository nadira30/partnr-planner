# Episode Export Methods

The episode visualizer now supports **three different export methods** with **two file formats**.

## Export Methods

### 1. Client-Side Download (default)
- **URL**: `/api/episode/<id>/export-config?download=client`
- **How it works**: Server sends JSON with metadata, JavaScript creates blob and triggers download
- **Use case**: Default browser download via UI
- **Response**: JSON with `episode` and `metadata` fields
- **Formats**: JSON only (browser can't easily compress to .gz)

### 2. Direct Download
- **URL**: `/api/episode/<id>/export-config?download=direct&format=<json|gz>`
- **How it works**: Server creates file in memory and sends via `send_file()`
- **Use case**: Clean direct download without JavaScript, supports .gz compression
- **Response**: Binary file stream
- **Formats**: JSON and GZ

### 3. Server-Side Save
- **URL**: `/api/episode/<id>/export-config?download=server&path=<optional_path>`
- **How it works**: Server saves files to disk and returns file paths
- **Use case**: Automation, scripts, batch processing
- **Response**: JSON with `paths` and `metadata` fields
- **Formats**: Both JSON and GZ saved simultaneously
- **Default location**: `visualization/data/episode_<id>_modified_<timestamp>.json`

## File Formats

### JSON (.json)
- Human-readable, easy to edit
- ~6KB for typical episode
- Specify with `format=json`

### Compressed (.json.gz)
- Gzip compressed JSON
- ~1.4KB for typical episode (77% smaller)
- Specify with `format=gz`
- **Only available with direct and server methods**

## Usage Examples

### From Browser UI
1. Add objects to episode
2. Select format (JSON or Compressed)
3. Select method (Client-side, Direct, or Server)
4. Click "Download Modified Episode"

### From Command Line

```bash
# Client method - get JSON response
curl "http://localhost:5002/api/episode/100/export-config?download=client"

# Direct download - JSON
curl "http://localhost:5002/api/episode/100/export-config?download=direct&format=json" \
  -o episode_100.json

# Direct download - Compressed
curl "http://localhost:5002/api/episode/100/export-config?download=direct&format=gz" \
  -o episode_100.json.gz

# Server save - both formats
curl "http://localhost:5002/api/episode/100/export-config?download=server"

# Server save - custom path
curl "http://localhost:5002/api/episode/100/export-config?download=server&path=/tmp/my_episode"
```

### From Python Script

```python
import requests

# Client method
response = requests.get("http://localhost:5002/api/episode/100/export-config?download=client")
data = response.json()
episode = data['episode']
metadata = data['metadata']

# Direct download
response = requests.get("http://localhost:5002/api/episode/100/export-config?download=direct&format=gz")
with open("episode_100.json.gz", "wb") as f:
    f.write(response.content)

# Server save
response = requests.get("http://localhost:5002/api/episode/100/export-config?download=server")
paths = response.json()['paths']
print(f"Saved to: {paths['json']} and {paths['gz']}")
```

## Additional Endpoint

### Export Raw Data
- **URL**: `/api/episode/<id>/export-data`
- **Returns**: Raw episode JSON without metadata wrapper
- **Use case**: When you only need the episode data, no metadata

## Comparison with Alternative Implementation

| Feature | Current Implementation | Alternative (3 endpoints) |
|---------|------------------------|---------------------------|
| **Endpoints** | 2 (unified + raw data) | 3 (export, export_data, export_gz) |
| **URL Structure** | RESTful with episode ID | Generic paths |
| **Client Download** | ✅ With metadata | ✅ Raw data only |
| **Direct Download** | ✅ JSON and GZ | ✅ GZ only |
| **Server Save** | ✅ Both formats + timestamp | ✅ Both formats |
| **Custom Path** | ✅ Via query param | ✅ Via POST body |
| **Metadata** | ✅ Objects added count | ❌ No metadata |
| **Flexibility** | Single unified endpoint | Multiple specialized endpoints |

## Best Practices

- **UI Users**: Use "Client-side download" for quick JSON export
- **Large Files**: Use "Direct download" with GZ format for compression
- **Automation**: Use "Server save" method to keep files on disk
- **Integration**: Use "Client method" to get data + metadata in one call
