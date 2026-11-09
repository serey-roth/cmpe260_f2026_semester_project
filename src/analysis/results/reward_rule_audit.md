# WebShop reward rule audit

Each case runs WebShop's own matching code. **FALSE MATCH** = the reward gives credit a human would not.

## Options (fuzzy match, token_set_ratio > 85)

| Goal option | Purchased option | Reward counts it | Should count | Verdict |
| --- | --- | --- | --- | --- |
| 6 ounce | 16 ounce | yes | no | FALSE MATCH |
| pack of 2 | pack of 12 | yes | no | FALSE MATCH |
| 1 count | 12 count | yes | no | FALSE MATCH |
| small | x-small | yes | no | FALSE MATCH |
| large | x-large | yes | no | FALSE MATCH |
| black | black red | yes | no | FALSE MATCH |
| blue | navy blue | yes | no | FALSE MATCH |
| king | california king | yes | no | FALSE MATCH |
| black | black | yes | yes | ok |
| medium | medium | yes | yes | ok |
| red | black | no | no | ok |

## Attributes (substring match in title, bullet points or description)

| Goal attribute | Field | Product text | Reward counts it | Should count | Verdict |
| --- | --- | --- | --- | --- | --- |
| red | Title | Reduced Fat Snack Crackers | yes | no | FALSE MATCH |
| long | Title | Belong Wall Art Print | yes | no | FALSE MATCH |
| cotton | Description | Made of polyester, not cotton. | yes | no | FALSE MATCH |
| machine washable | Description | Not machine washable. Dry clean only. | yes | no | FALSE MATCH |
| gluten free | Description | Contains wheat. Not gluten free. | yes | no | FALSE MATCH |
| gluten free | Title | Gluten Free Almond Crackers | yes | yes | ok |
| cotton | Description | 100% cotton fabric. | yes | yes | ok |

**13 of 14 cases that should not match were counted as matches.**
