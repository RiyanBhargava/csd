"""Canonical command line; no legacy architecture entry points."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="CSD: Cooperative Sign Decoder")
    commands = parser.add_subparsers(dest="command", required=True)
    campaign = commands.add_parser("campaign", help="Run the five-hour budgeted CSD study")
    campaign.add_argument("--name", default="csd_current")
    campaign.add_argument("--config", default="configs/study.json")
    campaign.add_argument("--stage", choices=("all", "calibrate", "evaluate"), default="all")
    campaign.add_argument("--dry-run", action="store_true")
    report = commands.add_parser("report", help="Summarize completed panels and list missing runs")
    report.add_argument("--name", default="csd")
    smoke = commands.add_parser(
        "smoke", help="Synthetic integration; optional real-model GPU check"
    )
    smoke.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.command == "campaign":
        if args.dry_run:
            from .campaign import run

            run(args.name, args.config, args.stage, True)
        else:
            from .budget import run_campaign

            run_campaign(args.name, args.config, args.stage)
    elif args.command == "report":
        from .campaign import report

        report(args.name)
    else:
        from .smoke import run

        run(args.gpu)


if __name__ == "__main__":
    main()
