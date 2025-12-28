# Working with EMDB

This section describes recommended practices for accessing and working with data from the Electron Microscopy Data Bank (EMDB), with a focus on scalable, reliable, and reproducible workflows.

---

## Recommended Access Methods

For large-scale data retrieval from EMDB, **FTP-based access is strongly recommended** over HTTPS.  
FTP provides significantly better stability and throughput, especially when downloading large numbers of entries or full-resolution density maps.
EMDB maintains multiple geographically distributed mirrors. Users should select the mirror closest to their location to maximize download performance.

## Recommended Mirror for Asia-Pacific Users

For users located in East Asia (e.g. China, Japan, Korea), the **PDBj mirror** provides substantially improved stability and download speed compared to the primary EBI servers.

- **FTP access:** `ftp://ftp.pdbj.org/pub/emdb/structures/` (also accessible via HTTP: [https://files.pdbj.org/pub/emdb/structures/](https://files.pdbj.org/pub/emdb/structures/))
- **Web interface:** [https://pdbj.org/emnavi/](https://pdbj.org/emnavi/)

### Example: Downloading Entry EMD-0153

```bash
# Using wget
wget https://ftp.pdbj.org/pub/emdb/structures/EMD-0153/map/emd_0153.map.gz
```

## Programmatic Dataset Selection via EMDB Search API

Before downloading large datasets, we strongly recommend using the **EMDB Search API** to programmatically identify entries of interest.  
This enables efficient filtering based on metadata and avoids unnecessary data transfer.

Common filtering criteria include:

- Resolution range  
- Molecular weight  
- Structure determination method (e.g. single-particle cryo-EM)  
- Entry status  

### Example query (CSV output)

```text
https://www.ebi.ac.uk/emdb/api/search/database:EMDB AND current_status:[* TO *] AND assembly_molecular_weight:{0 TO 10000000] AND resolution:[0 TO 3} AND structure_determination_method:"singleparticle"?rows=1000000&wt=csv&download=false&fl=emdb_id,structure_determination_method,resolution,resolution_method,assembly_molecular_weight,cpx_name
```

This interface allows users to:
- programmatically retrieve metadata,
- construct reproducible dataset selection pipelines,
- selectively download only relevant EMDB entries.

---

## Acknowledgments

We thank Kyle Morris for his contributions and insights regarding EMDB data access and workflow recommendations.
