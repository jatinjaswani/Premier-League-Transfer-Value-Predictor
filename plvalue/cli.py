"""Command-line interface:  python -m plvalue <command>"""
import argparse

from . import api, model, transfermarkt


def main(argv=None):
    ap = argparse.ArgumentParser(prog="plvalue", description="Premier League market-value predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="check which football-data.org endpoints your plan can reach")
    sub.add_parser("fetch", help="download PL + World Cup stats -> data/processed/stats.csv")
    sub.add_parser("build", help="match players to Transfermarkt -> values.csv + tm_features.csv")

    tp = sub.add_parser("train", help="cross-validate, run the ablation study, save the model")
    tp.add_argument("--model", default="linear", choices=["linear", "ridge", "gbm"])
    tp.add_argument("--use-current", action="store_true",
                    help="include 2026/27 stats (off by default: values predate that season)")

    ep = sub.add_parser("errors", help="list the players the model gets most wrong")
    ep.add_argument("--top", type=int, default=12)

    pp = sub.add_parser("predict", help="estimate a player's value")
    pp.add_argument("--name", required=True)

    sub.add_parser("readme", help="paste reports/results.md into README.md")

    a = ap.parse_args(argv)
    {
        "probe": api.probe,
        "fetch": api.fetch,
        "build": transfermarkt.build,
        "train": lambda: model.train(a.model, a.use_current),
        "errors": lambda: model.errors(a.top),
        "predict": lambda: model.predict(a.name),
        "readme": model.update_readme,
    }[a.cmd]()


if __name__ == "__main__":
    main()
