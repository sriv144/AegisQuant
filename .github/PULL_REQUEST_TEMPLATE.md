## Summary

<!-- One or two sentences on what this PR does and why. -->

## Type of change

- [ ] `feat`     new feature
- [ ] `fix`      bug fix
- [ ] `perf`     performance improvement
- [ ] `refactor` non-functional code change
- [ ] `docs`     documentation only
- [ ] `test`     tests only
- [ ] `chore`    build / config / tooling
- [ ] `ci`       CI changes

## Test plan

- [ ] `make test` is green locally
- [ ] `make lint` is green locally
- [ ] Manually verified the affected path (describe how below)

<!-- describe manual verification here -->

## Live-trading safety checklist

- [ ] No new code path bypasses `LongOnlyRule` or `MaxPositionRule`.
- [ ] No real broker API key is committed; new env vars added to `.env.example`.
- [ ] If touching position sizing or order routing, a paper-trading dry run was performed.

## Related

Closes #...
