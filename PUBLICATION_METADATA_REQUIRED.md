# Publication and repository metadata required

Before public upload or journal submission, accountable humans must supply
and approve:

- repository owner/organization and final URL;
- author names, order, affiliations, ORCIDs, and corresponding email;
- CRediT roles, funding, grants, conflicts, and acknowledgments;
- copyright holder and explicit license approval;
- third-party dataset, model, and derived-artifact rights;
- journal-required AI-assistance disclosure;
- originality and concurrent-submission declarations; and
- final proofreading, categories/EDICS, and submission form selections.

The technical package intentionally does not invent these facts. After
approval, create `LICENSE` from `LICENSE.template`, add valid `CITATION.cff`,
update the human-only status text, and run
`refresh_release_manifest.py --confirm-accountable-metadata`.
