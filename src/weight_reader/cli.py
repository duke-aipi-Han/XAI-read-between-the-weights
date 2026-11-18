"""Prepare synthetic datasets and train the initial classifier checkpoints."""
import argparse
import json
from pathlib import Path

import torch

from .data import make_split, make_splits
from .editing import edit_weights, validation_metrics
from .model import WeightReaderMLP, calibration_model
from .probe import probe_round, start_session
from .train import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train'):
        command = sub.add_parser(name)
        command.add_argument('--config', default='experiment.json')
        command.add_argument('--output-dir', default='.', help='project output root')
    probe = sub.add_parser('inspect', help='return ordinary logits for exploratory inputs')
    probe.add_argument('--checkpoint', required=True)
    probe.add_argument('--inputs', required=True, help='JSON file containing an array of four-coordinate inputs')
    edit = sub.add_parser('edit', help='write a scaled parameter slice to a new checkpoint')
    edit.add_argument('--checkpoint', required=True)
    edit.add_argument('--edit', required=True, help='JSON with parameter, index, and scale')
    edit.add_argument('--output', required=True)
    validate = sub.add_parser('validate', help='score a checkpoint on the development validation split')
    validate.add_argument('--checkpoint', required=True)
    validate.add_argument('--config', default='experiment.json')
    session = sub.add_parser('start-probe-session', help='create a fixed or adaptive probing session')
    session.add_argument('--session-id', required=True)
    session.add_argument('--condition', choices=('fixed_probes', 'adaptive_probes'), required=True)
    session.add_argument('--checkpoint', required=True)
    session.add_argument('--config', default='experiment.json')
    session.add_argument('--output-dir', default='.')
    relay = sub.add_parser('probe', help='run or replay one budgeted probe round')
    relay.add_argument('--session-id', required=True)
    relay.add_argument('--round', type=int, required=True)
    relay.add_argument('--inputs', help='adaptive JSON file with inputs and optional hypothesis')
    relay.add_argument('--output-dir', default='.')
    args = parser.parse_args()
    if args.command in ('start-probe-session', 'probe'):
        try:
            if args.command == 'start-probe-session':
                config = json.loads(Path(args.config).read_text())
                start_session(args.output_dir, args.session_id, args.checkpoint, args.condition, config)
                print(f'Created probe session {args.session_id}')
            else:
                payload = json.loads(Path(args.inputs).read_text()) if args.inputs else None
                result = probe_round(args.output_dir, args.session_id, args.round, payload)
                print(json.dumps(result, allow_nan=False))
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        return
    torch.set_num_threads(1)
    if args.command in ('inspect', 'edit', 'validate'):
        model = WeightReaderMLP()
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        if args.command == 'inspect':
            inputs = torch.tensor(json.loads(Path(args.inputs).read_text()), dtype=torch.float32)
            if inputs.ndim != 2 or inputs.shape[1] != 4 or len(inputs) == 0 or not torch.isfinite(inputs).all() or (inputs.abs() > 1).any():
                parser.error('inputs must be a nonempty array of finite four-vectors in [-1, 1]')
            with torch.no_grad():
                logits = model(inputs).squeeze(1).tolist()
            print(json.dumps({'inputs': inputs.tolist(), 'logits': logits}))
        elif args.command == 'edit':
            output = Path(args.output)
            if output.exists():
                parser.error('output already exists; choose a new path')
            spec = json.loads(Path(args.edit).read_text())
            edited = edit_weights(model, **spec)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also prevents overwrites through an existing link.
            with output.open('xb') as file:
                torch.save(edited.state_dict(), file)
            print(f'Saved edited checkpoint to {output}')
        else:
            config = json.loads(Path(args.config).read_text())
            print(json.dumps(validation_metrics(model, make_split(**config['splits']['validation'])), indent=2))
        return
    config = json.loads(Path(args.config).read_text())
    root = Path(args.output_dir)
    splits = make_splits(config)
    torch.set_num_threads(1)
    if args.command == 'prepare':
        paths = [root / 'private' / 'splits.pt', root / 'checkpoints' / 'calibration.pt']
        if any(path.exists() for path in paths):
            parser.error('output already exists; choose a new --output-dir to avoid overwriting')
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(splits, paths[0])
        torch.save(calibration_model().state_dict(), paths[1])
        print('Saved private datasets and the calibration checkpoint.')
    else:
        split_path = root / 'private' / 'splits.pt'
        if not split_path.exists():
            parser.error('run prepare first to create the private datasets')
        saved_splits = torch.load(split_path, map_location='cpu', weights_only=True)
        if set(saved_splits) != set(splits) or any(
            not all(torch.equal(a, b) for a, b in zip(saved_splits[name], splits[name]))
            for name in splits
        ):
            parser.error('saved datasets differ from config; prepare a new output directory')
        splits = saved_splits
        paths = [root / 'checkpoints' / f'trained_seed_{seed}.pt' for seed in config['training']['seeds']]
        metrics_path = root / 'results' / 'training_pilot.json'
        if any(path.exists() for path in [*paths, metrics_path]):
            parser.error('output already exists; choose a new --output-dir to avoid overwriting')
        metrics = []
        for seed, path in zip(config['training']['seeds'], paths):
            model, record = train_model(splits['train'], splits['validation'], config['training'], seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            metrics.append(record)
            print(json.dumps(record), flush=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
