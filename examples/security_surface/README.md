# Artifacts beside the policy (illustrative; deliberately insecure)

Use these with `examples/security/institution_access_policy.txt` to see the RSI
loop read code and procedures against a written policy:

```powershell
deepthink rsi --input examples/security/institution_access_policy.txt --repo examples/security_surface/repo --social examples/security_surface/social_profile.json --offline
```

* `repo/` is a tiny fake service with planted defects: a hard-coded database
  password, TLS verification off, debug mode, string-built SQL, an HR database
  open to the internet, a public bucket of HR exports, remote desktop switched
  on, and a container that runs as root. The secret is a placeholder, not a
  real credential.
* `social_profile.json` describes roles and procedures only. It has an
  `authorization` block, and its `people` list shows that individual profiles
  are dropped, not analysed.
