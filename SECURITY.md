# Security Policy

This is a research repository. It trains models, writes reports, and calls two external
APIs; it does not run a service, handle user accounts, or process untrusted input in
production. The realistic risk surface is credentials and supply chain.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/MD-ALL-SHAHRIA/pulseair/security/advisories/new).
Please don't open a public issue for anything exploitable.

Expect a first response within about a week. This is maintained alongside a degree, not
full-time.

## Found a leaked credential in this repo?

Report it privately, as above. Every commit is scanned for credential patterns before
it's pushed, and no key has been committed — but if one ever slips through, tell us
before anyone else.

## Handling keys yourself

- Keys go in `.env`, which is gitignored and must stay that way. Only variable *names*
  belong in `.env.example`.
- `GEMINI_API_KEY` and `OPENAQ_API_KEY` are both optional; the code degrades to a
  rule-based template and a cached survey respectively when they're absent.
- Never print a key in report output, log lines, or a traceback.

## Dependencies

`requirements.lock.txt` pins the exact resolved set used to produce the committed
results, which is what reproducibility needs — but a lockfile ages. For everyday use
install from `requirements.txt` and let pip resolve current patched versions within the
stated ranges.

Note that `numpy` is pinned `<2.0` because parts of the SDV stack still build against
numpy 1.x. If that pin ever blocks a security update, break the pin and the SDV path,
not the other way around.
