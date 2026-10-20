### CI: the private-marker guard reports as `private-marker-guard`

The guard's job reports under its own name instead of the generic `guard`, and only pull_request runs carry it;
push runs report as `private-marker-guard (push scan)`. Every branch push is still scanned.
