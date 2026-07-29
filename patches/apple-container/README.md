# Apple container patches

Keep upstream `third_party/apple-container` clean by default. If an integration change becomes necessary:

1. base it on pinned commit `14233cee65486c1ada2b82403c17d1236a9176c2`;
2. document the reason, exact files, types, and functions in the patch commit message/header;
3. store a reviewable `.patch` file here;
4. apply it with `scripts/apply-apple-container-patches.sh`;
5. reverse it with `scripts/reverse-apple-container-patches.sh` before updating the submodule.

No Apple-container patch is required for the current local protocol prototype.
