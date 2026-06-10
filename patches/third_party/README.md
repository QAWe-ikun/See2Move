# Third-Party Patches

The top-level repository tracks `third_party/vln-ce` and
`third_party/smartway-code` as submodules. Local compatibility edits are stored
as patch files here so the third-party histories remain clean.

Apply after cloning submodules:

```bash
cd third_party/vln-ce
git apply ../../patches/third_party/vln-ce-opencv-headless.patch

cd ../smartway-code
git apply ../../patches/third_party/smartway-openai-env-key.patch
```

Current pinned submodule commits:

- `third_party/vln-ce`: `729d141b2ee10628061ada74dd3a5b9f70faeba5`
- `third_party/smartway-code`: `daa2dd856872727832b4b07cd3a09db34cf211d4`
