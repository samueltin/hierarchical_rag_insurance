# Demo Questions

Ten questions, each with two follow-ups that only make sense in context. Every
one was run against the live index — all 30 return a grounded answer.

**Ask the follow-ups without repeating the subject.** They are written as
ellipsis and pronouns ("What about in Europe?", "Can I protect it?") precisely
because that is what breaks naive RAG: the raw text has none of the words the
search needs. Query condensation rewrites each one into a standalone question
before searching, and the chat UI shows the rewrite under the message as
*🔍 searched as: …* — worth pointing at during the demo.

Sources: `breakdown_policy_booklet.pdf`, `car-insurance-policy-booklet.pdf`,
`important-information.pdf`.

---

### 1. Roadside cover — basic retrieval

> **What is covered under Section A Roadside?**
> - What about in Europe?
> - Does it cover my caravan as well?

Answers from `breakdown_policy_booklet`. The first follow-up is the clearest
demonstration of condensation: *"What about in Europe?"* alone retrieves almost
nothing, condensed it finds `Section E2: Roadside assistance in Europe`.

### 2. Onward travel — limits held in a list

> **What alternative transport is available under Section D Onward Travel?**
> - Is overnight accommodation included too?
> - What is the limit per person?

The last question is pure ellipsis — "limit" of what, for whom, is carried
entirely by the conversation. Answers £150 per person from
`3. Overnight accommodation`.

### 3. Definitions — a defined term

> **What does 'beyond economical repair' mean?**
> - Does the same apply if it happens in Europe?
> - Will the RAC still get my vehicle home?

The primary retrieves the definition from **both** policy booklets, so the
answer separates them by document — a good moment to point out cross-document
attribution.

### 4. Cancellation fees — a table

> **What fees are charged if I cancel my car insurance policy?**
> - What if I cancel within the first 14 days?
> - Do I still get a refund if I have already claimed?

The fee table is split across three child chunks with the `£` amounts separated
from their row labels, but the **parent** holds the whole table, so the figures
come back correct (£28 / £56 / £11.20). This is the parent-child design earning
its keep — worth calling out.

### 5. No claim discount

> **How does the no claim discount work on my car insurance?**
> - What happens to it if I have an accident that is not my fault?
> - Can I protect it?

Two consecutive "it" references, resolved from context.

### 6. Glass cover — a cross-section link

> **Am I covered for windscreen and glass damage?**
> - Does claiming for that affect my no claim discount?
> - What if the windscreen can be repaired instead of replaced?

The first follow-up jumps from Section 5 to Section 4 — the condensed query has
to carry both concepts.

### 7. Courtesy car

> **When do I get a courtesy car?**
> - How long can I keep it?
> - What if my car is written off?

### 8. Driving abroad

> **Can I drive my car in Europe under Section 6 Foreign use?**
> - How long can I stay there?
> - What if I need longer than that?

Answers 90 days per trip, six months per period of insurance, and that an
extension must be agreed and paid for.

### 9. Data protection — the third document

> **How is my personal information used and shared by insurers?**
> - Who do they share it with?
> - What are they doing to prevent fraud?

Answers from `important-information`, showing the corpus is not just the two
policy booklets.

### 10. Complaints — the strongest cross-document case

> **How do I make a complaint?**
> - What if I am not happy with the outcome?
> - Who do I contact about a breakdown complaint?

Every document has a "Complaints" section with a **different** route. The answer
groups them by policy rather than merging them into one list. Ask the last
follow-up to show the search narrowing to the breakdown booklet on its own.

---

## Suggested demo order

1. **Q1** — basic answer, citation panel, groundedness badge.
2. **Q1 follow-up** — point at *🔍 searched as:* to show condensation.
3. **Q4** — the fee table; parent-child chunking.
4. **Q10** — cross-document attribution.
5. **Q9** — third document, and the document scope selector in the sidebar.

Switch users in the sidebar to show that conversations are per-user and persist,
then start a new chat to show history is stored, not session state.

## Questions that deliberately fail

Useful for showing honest behaviour rather than confident invention. All four
were checked — each answers "I don't have enough information to answer that."

> **What is the excess on my policy?** — the excess lives in the schedule, which
> is not part of the indexed corpus. The answer says so, and points at the
> schedule rather than inventing a figure.

> **What is my policy number?** — personal data the booklets cannot contain.

> **How much will my premium be next year?** — not in any booklet.

> **Does this policy cover my motorbike?** — outside the scope of a motor policy
> booklet, and a good test that it does not answer from general knowledge.
