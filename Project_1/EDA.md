# EDA: what the data looks like (computed by `eda.py`, not hand-written)

## 1. Size

| split | docs | CUAD docs | ExtractBench docs | fields | fields with a value | fields absent (gold null) |
|---|---|---|---|---|---|---|
| dev | 540 | 381 | 159 | 22115 | 9372 (42.4%) | 12743 (57.6%) |
| test_public | 176 | 129 | 47 | 6962 | 2865 (41.2%) | 4097 (58.8%) |

**Ceiling on savings.** Only 42.1% of all fields have a value. An absent field can never be 'booked' (a true null costs one review, saves nothing), so even a perfect system can save at most 42.1% of the manual cost. Every wrongly accepted value costs 100 reviews, so the realistic goal is *high precision on a modest, safe slice*.

## 2. Per source

| source | fields | absent | fields/doc (median / max) |
|---|---|---|---|
| cuad | 20910 | 67.9% | 41 / 41 |
| extractbench | 8167 | 32.2% | 27 / 159 |

## 3. CUAD (contracts, 41 clause types, span answers)

- Contract length (chars): median 33,143, p90 122,688, max 338,211.
- **28.4% of contracts are longer than 60,000 characters**, the cut-off used by the starter, so a naive 'paste the contract into the prompt' extractor silently never sees the end of the document (where termination, liability and governing-law clauses often sit). A small model with a 32K-token window is worse off. -> we retrieve candidate paragraphs per clause instead of truncating.

Absent-rate per clause type (a model that 'finds' a clause here is hallucinating):

| clause | absent | clause | absent | clause | absent |
|---|---|---|---|---|---|
| Document Name | 0% | Parties | 0% | Agreement Date | 8% |
| Governing Law | 14% | Expiration Date | 19% | Effective Date | 24% |
| Anti-Assignment | 27% | Cap On Liability | 46% | License Grant | 50% |
| Audit Rights | 58% | Termination For Convenience | 64% | Post-Termination Services | 64% |
| Exclusivity | 65% | Renewal Term | 65% | Insurance | 67% |
| Revenue/Profit Sharing | 67% | Minimum Commitment | 68% | Non-Transferable License | 73% |
| Ip Ownership Assignment | 76% | Change Of Control | 76% | Non-Compete | 77% |
| Notice Period To Terminate Renewal | 78% | Uncapped Liability | 78% | Covenant Not To Sue | 80% |
| Rofr/Rofo/Rofn | 83% | Volume Restriction | 84% | Competitive Restriction Exception | 85% |
| Warranty Duration | 85% | Irrevocable Or Perpetual License | 86% | Liquidated Damages | 88% |
| Affiliate License-Licensee | 88% | No-Solicit Of Employees | 88% | Joint Ip Ownership | 91% |
| Non-Disparagement | 93% | No-Solicit Of Customers | 93% | Third Party Beneficiary | 94% |
| Most Favored Nation | 95% | Affiliate License-Licensor | 95% | Unlimited/All-You-Can-Eat-License | 97% |
| Price Restrictions | 97% | Source Code Escrow | 97% |

Always-present: Document Name. Almost-never-present (>=90% absent): Joint Ip Ownership, Non-Disparagement, No-Solicit Of Customers, Third Party Beneficiary, Most Favored Nation, Affiliate License-Licensor, Unlimited/All-You-Can-Eat-License, Price Restrictions, Source Code Escrow.

- Gold answers are long: median 274 chars, p90 1280 chars; scoring is token-F1 >= 0.8, so the model must copy the *whole* clause, not a keyword.

## 4. ExtractBench (forms and scans)

| value kind | fields | share |
|---|---|---|
| absent | 2632 | 32.2% |
| text | 2626 | 32.2% |
| checkbox=false | 978 | 12.0% |
| integer | 882 | 10.8% |
| checkbox=true | 373 | 4.6% |
| nested json (exact match) | 366 | 4.5% |
| decimal | 310 | 3.8% |

- **An empty checkbox is not null**: it is the string `false` (978 such fields vs 373 `true`). A system that returns null for an unticked box wastes a free, easy saving; one that reads a scan badly and says `false` for a ticked box books a wrong value.
- Nested objects/arrays (366 fields) are scored all-or-nothing, so one wrong cell makes the whole value wrong. These are the fields we would never automate.

Capture-condition / challenge tags (docs):

`challenge:T3.a`: 146, `domain:D1`: 87, `perception:P2`: 84, `domain:D2`: 70, `perception:P3`: 34, `challenge:T1.a`: 25, `domain:D3`: 18, `domain:D5`: 15, `challenge:T3.b`: 15, `challenge:T3.e`: 14, `challenge:T1.b`: 12, `challenge:T1.d`: 12

- Schema size (JSON chars): median 16,589, max 567,854. Some forms have 159 fields, which does not fit one reliable prompt for a small model: we chunk the schema.

## 5. Reference points

- 'Never answer' saves 0%. 'Most common value' (3_baseline.py) is the bar a real system must clear.
- Break-even precision at review cost 1 and wrong-value cost 100 is **99.0%**: accepting costs 100 × (1 − p), while review costs 1, so acceptance breaks even at p = 0.99 and pays off only above it. This number, not accuracy, drives the whole design of ProofGuard.
