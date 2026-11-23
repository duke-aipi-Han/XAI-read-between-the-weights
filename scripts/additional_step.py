"""Send one explicitly authorized, journaled additional-reviewer API turn."""
import json, sys
from pathlib import Path
from weight_reader.remote_additional import remote_step
root=Path('private/studies/repair_additional')
config=json.loads(Path('remote_config.json').read_text())
print(json.dumps(remote_step(root,sys.argv[1],'.env',config),indent=2),flush=True)
