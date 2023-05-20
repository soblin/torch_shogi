import glob
import os
import sys
from argparse import ArgumentParser

import cshogi
import numpy as np
from cshogi import CSA
from sklearn.model_selection import train_test_split


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("csa_dir")
    parser.add_argument("hcpe_train")
    parser.add_argument("hcpe_test")
    parser.add_argument("--filter_moves", type=int, default=50)
    parser.add_argument("--filter_rating", type=int, default=3500)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    csa_file_list = glob.glob(
        os.path.join(args.csa_dir, "**", "*.csa"), recursive=True
    )
    files_train, files_test = train_test_split(
        csa_file_list, test_size=args.test_ratio
    )
    hcpes = np.zeros(
        1024, cshogi.HuffmanCodedPosAndEval
    )  # もしかして将棋は高々1024手で終了するから？
    f_train = open(args.hcpe_train, "wb")
    f_test = open(args.hcpe_test, "wb")
    board = cshogi.Board()

    # 訓練とテストをくっつけて一気にzipで回している
    for file_list, f in zip([files_train, files_test], [f_train, f_test]):
        kif_num = 0
        position_num = 0
        for filepath in file_list:
            for kif in CSA.Parser.parse_file(filepath):
                # 投了，千日手，宣言勝ちで終了したものは除外
                if kif.endgame not in ("%TORYO", "%SENNICHITE", "%KACHI"):
                    continue
                # すぐに終了したものは除外
                if len(kif.moves) < args.filter_moves:
                    continue
                # レートが低いものは除外
                if (
                    args.filter_rating > 0
                    and min(kif.ratings) < args.filter_rating
                ):
                    continue

                board.set_sfen(kif.sfen)
                p = 0
                try:
                    for i, (move, score, comment) in enumerate(
                        zip(kif.moves, kif.scores, kif.comments)
                    ):
                        if not board.is_legal(move):
                            raise Exception()
                        hcpe = hcpes[p]  # 参照
                        p += 1
                        board.to_hcp(hcpe["hcp"])
                        # 16bitに収まるようにクリッピング
                        eval = min(32767, max(score, -32767))
                        # 手番側の評価値にする
                        hcpe["eval"] = (
                            eval if board.turn == cshogi.BLACK else -eval
                        )
                        # 指し手の32bit数値を16bitに切り捨てる
                        hcpe["bestMove16"] = cshogi.move16(move)
                        # 勝敗結果
                        hcpe["gameResult"] = kif.win
                        board.push(move)  # 盤面を進める
                except Exception as e:
                    print(f"e{e}, skip {filepath}")
                    continue

                if p == 0:
                    continue

                hcpes[:p].tofile(f)  # p手までの全てを書き出す

                kif_num += 1
                position_num += p

        print(f"kif_num = {kif_num}")
        print(f"positon_num = {position_num}")

    print(len(files_train), len(files_test))


if __name__ == "__main__":
    main(sys.argv)
