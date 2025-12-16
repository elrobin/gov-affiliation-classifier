# Gov Affiliation Classifier

A Python utility for classifying institutional affiliations using a hybrid approach combining:
- **ROR (Research Organization Registry)** knowledge base for fast matching
- **Rule-based classification** for clear cases (universities, funders, teaching hospitals)
- **LM Studio** (local LLM) for complex or ambiguous cases

The classifier determines:
- Organization type (`org_type`)
- Government level (`gov_level`) - primarily defined for USA (federal, state, local), extensible to other countries
- Local government type (`gov_local_type`)
- Research mission category (`mission_research_category`)
- Research mission binary flag (`mission_research`)

**Note:** This tool is designed for research analysis and does not reproduce official administrative classifications.

## Features

### 🚀 Fast-Track Classification
The system uses a two-stage approach for efficiency:

1. **Rule-based classification** (fast): For clear cases identified via ROR matching:
   - Universities with education type in ROR
   - Research funding organizations (Enablers)
   - Teaching hospitals with academic links

2. **LLM classification** (when needed): For ambiguous cases, complex organizations, or when ROR data is unavailable.

3. **Prompt & parsing safeguards:** The prompt now enforces ROR-aligned research mission (mission_research=1 when a valid ROR match exists and mission_research_category limited to AcademicResearch/AppliedResearch/Enabler) and the client extracts JSON robustly even if the model wraps it in ```json fences.

### 📊 Output Fields

The classifier produces a CSV with the following fields:

**Classification fields:**
- `org_type`: One of `supranational_organization`, `government`, `university`, `research_institute`, `company`, `ngo`, `hospital`, `other`
- `gov_level`: One of `federal`, `state`, `local`, `unknown`, `non_applicable`
- `gov_local_type`: One of `city`, `county`, `other_local`, `unknown`, `non_applicable`
- `mission_research_category`: One of `NonResearch`, `Enabler`, `AppliedResearch`, `AcademicResearch`
- `mission_research`: Binary (0 or 1), derived from `mission_research_category`
  - `1` if category is `AppliedResearch` or `AcademicResearch`
  - `0` if category is `NonResearch` or `Enabler`

**ROR enrichment fields:**
- `ror_id`: ROR identifier if matched
- `ror_name`: Official ROR name
- `ror_types`: Comma-separated list of ROR types
- `ror_country_code`: Country code from ROR
- `ror_state`: State/region from ROR (if available)
- `ror_city`: City from ROR (if available)
- `ror_match_score`: Match confidence score (0.0-1.0)
- `suggested_org_type_from_ror`: Suggested organization type based on ROR types

## Current Scope (v1.0)

### Geographic Scope

In v1.0, `gov_level` is primarily validated for U.S. affiliations (federal / state / local). For other countries, values may be `unknown` or `non_applicable`. The design is extensible to other countries, but administrative taxonomies are not yet country-specific.

### Academic Use

**Important:** This tool is designed for **research analysis purposes** and does not aim to reproduce official administrative classifications. The taxonomy and classifications are intended for academic and research use, not for official government or legal purposes.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have:
   - A local LM Studio instance running with an OpenAI-compatible API
   - ROR dump file (default: `v1.74-2025-11-24-ror-data/v1.74-2025-11-24-ror-data.json`)

## Usage

### Basic Usage

```bash
python main.py --input input.csv --output output.csv
```

### With Custom ROR Path

```bash
python main.py --input input.csv --output output.csv --ror-path path/to/ror-data.json
```

### Input CSV Format

The input CSV must contain the following columns:
- `afid`: Unique identifier for each affiliation
- `affiliation`: The affiliation string to classify
- `country_code`: ISO country code (e.g., `usa`, `gbr`, `fra`)

Example:
```csv
afid,affiliation,country_code
1,"Harvard University",usa
2,"National Science Foundation",usa
3,"Ministry of Health",fra
```

### Output CSV Format

The output CSV contains all input columns plus the classification and ROR enrichment fields listed above.

## Configuration

Environment variables (optional) may be defined in a `.env` file:

- `LM_STUDIO_BASE_URL` (default: `http://localhost:1234/v1/chat/completions`)
- `LM_STUDIO_MODEL_NAME` (default: `local-model`)
- `LM_STUDIO_TIMEOUT` in seconds (default: `60`)

## Architecture

### Components

1. **`ror_knowledge.py`**: ROR knowledge base module
   - Loads and indexes ROR dump
   - Provides fuzzy matching against ROR organizations
   - Suggests organization types based on ROR metadata

2. **`lm_client.py`**: LLM client and rule-based classifier
   - `try_rule_based_classification()`: Fast-track classification for clear cases
   - `classify_affiliation()`: LLM-based classification for complex cases
   - System prompt with taxonomy definitions

3. **`main.py`**: Main orchestration script
   - Reads input CSV
   - Loads ROR knowledge base
   - Applies rule-based classification when possible
   - Falls back to LLM when needed
   - Writes enriched output CSV

### Classification Flow

```
For each affiliation:
  1. Match against ROR knowledge base
  2. Try rule-based classification:
     - If clear match → use rule-based result (skip LLM)
     - If ambiguous → proceed to step 3
  3. Call LLM with ROR context (if available)
  4. Validate and normalize LLM response
  5. Write results to CSV
```

### Rule-Based Classification Rules

The system applies conservative rules only for clear cases:

**Rule A - Universities:**
- ROR type includes `education`
- Name contains university indicators (university, college, institute of technology, etc.)
- Excludes consortia and associations
- Result: `org_type=university`, `mission_research_category=AcademicResearch`

**Rule B - Funders:**
- ROR type includes `funder`
- Name contains funding indicators (foundation, research council, etc.)
- Determines government vs NGO based on name patterns
- Result: `mission_research_category=Enabler`

**Rule C - Teaching Hospitals:**
- ROR type includes `healthcare`
- Name contains academic hospital indicators (university hospital, academic medical center, etc.)
- Result: `org_type=hospital`, `mission_research_category=AcademicResearch`

**Important:** If any rule is ambiguous or doesn't apply, the system defaults to LLM classification.

## Research Mission Categories

The classifier uses a four-category taxonomy for research mission:

- **`NonResearch`** (code 0): Organizations with no explicit research mission
  - General-purpose government agencies, administrative offices
  - Hospitals/clinics without research role
  - Service providers implementing policy

- **`Enabler`** (code 1): Organizations that enable or fund research but don't primarily produce it
  - Research funding agencies (NSF, NIH HQ, research councils)
  - Foundations and philanthropies that fund research
  - University grant offices

- **`AppliedResearch`** (code 2): Organizations conducting mission-oriented or applied research
  - Government labs (EPA, USGS, NOAA research centers)
  - Public health services with analytic capabilities
  - R&D units in agencies or companies

- **`AcademicResearch`** (code 3): Organizations where research is a central formal mission
  - Universities and higher education institutions
  - Research institutes and centers
  - National labs with strong scientific mission
  - Academic medical centers linked to universities

The binary `mission_research` flag is automatically derived:
- `1` for `AppliedResearch` or `AcademicResearch`
- `0` for `NonResearch` or `Enabler`

**ROR prior (prompt rule):** when a valid ROR record is present, the prompt instructs the model to treat the organization as part of the research ecosystem, set `mission_research` = 1, and pick `mission_research_category` from `AcademicResearch`, `AppliedResearch`, or `Enabler` according to ROR role cues.

## Performance

The hybrid approach significantly reduces LLM calls:
- Clear cases (universities, funders, teaching hospitals) are classified instantly via rules
- Only ambiguous or complex cases require LLM inference
- Typical reduction: 30-50% fewer LLM calls depending on dataset composition

The system logs statistics:
```
Classification complete: 45 rule-based, 55 LLM calls
```

## Performance and Limitations

The classifier is designed for **local execution** with LM Studio. When running on CPU-only local LLMs, some complex batches may take several minutes to complete. The pipeline is designed to favor conservative inference and fallback mechanisms over aggressive parallelization.

**Important considerations:**
- In CPU-only setups, long-running batch inference may be slow
- Complex batches with many ambiguous cases may require several minutes per batch
- The system prioritizes robustness and correctness over speed
- Timeouts are set conservatively to accommodate slower inference on CPU

## Error Handling

- If ROR dump fails to load, the system continues without ROR matching
- If ROR matching fails for a specific affiliation, it falls back to LLM
- If LLM call fails, classification fields are set to `None` for that row
- All errors are logged for debugging

## Dependencies

- `pandas>=2.0`: CSV processing
- `requests>=2.31`: HTTP client for LM Studio API
- `python-dotenv>=1.0`: Environment variable management
- `rapidfuzz>=3.0`: Fuzzy string matching for ROR

## Future Improvements

Future versions may include checkpointing and resume capabilities for very large datasets, allowing the pipeline to recover from interruptions and continue processing from the last successful batch.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]
