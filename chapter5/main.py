import cshogi
import torch
import torch.nn as nn
import torch.nn.functional as F

MOVE_DIRECTION = [
    UP,
    UP_LEFT,
    UP_RIGHT,
    LEFT,
    RIGHT,
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    UP2_LEFT,
    UP2_RIGHT,
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE,
] = range(20)


FEATURES_NUM = len(cshogi.PIECE_TYPES) * 2 + sum(cshogi.MAX_PIECES_IN_HAND)

MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(cshogi.HAND_PIECES)
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81


def set_board_features(board, features):
    # features: (駒のchanel, 9, 9)の配列
    # features[i]のiは90ページにあるそれぞれの駒のenum値
    # features[0]は先手の歩の位置で右側の縦3列目に相当し，1が入る
    # features[14]は後手の歩の位置で左の縦3列目に相当し，1が入る
    # features[28]以降は持ち駒
    features.fill(0)

    if board.turn == cshogi.BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_hand)

    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, cshogi.MAX_PIECES_IN_HAND):
            features[i : i + num].fill(1)
            i += max_num


def make_move_label(move, color):
    if not cshogi.move_is_drop(move):
        to_sq = cshogi.move_to(move)
        from_sq = cshogi.move_from(move)

        if color == cshogi.WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_RIGHT
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_LEFT
            elif dir_x < 0:
                move_direction = UP_RIGHT
            else:
                move_direction = UP_LEFT
        elif dir_y == 0:
            if dir_x < 0:
                move_direction = RIGHT
            else:
                move_direction = LEFT
        else:
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x < 0:
                move_direction = DOWN_RIGHT
            else:
                move_direction = DOWN_LEFT

        if cshogi.move_is_promotion(move):
            move_direction += 10

    else:
        to_sq = cshogi.move_to(move)
        if color == cshogi.WHITE:
            to_sq = 80 - to_sq

        move_direction = len(
            cshogi.MOVE_DIRECTION
        ) + cshogi.move_drop_hand_piece(move)

    return move_direction * 81 + to_sq


def make_result(game_result, color):
    if color == cshogi.BLACK:
        if game_result == cshogi.BACK_WIN:
            return 1
        if game_result == cshogi.WHITE_WIN:
            return 0
    else:
        if game_result == cshogi.BLACK_WIN:
            return 0
        if game_result == cshogi.WHITE_WIN:
            return 1
    return 0.5


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )  # BatchNormalizationを続ける場合はバイアスは不要
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out + x)


class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.parameter.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=0, channels=192, fcl=256):
        super(PolicyValueNetwork, self).__init__()
        # 中間層
        self.conv1 = nn.Conv2d(
            in_channels=FEATURES_NUM,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(
            *[ResNetBlock(channels) for _ in range(blocks)]
        )

        # policy head
        self.policy_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=MOVE_PLANES_NUM,
            kernel_size=1,
            bias=False,
        )
        self.policy_bias = Bias(MOVE_LABELS_NUM)

        # value head
        self.value_conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=MOVE_PLANES_NUM,
            kernel_size=1,
            bias=False,
        )
        self.value_norm1 = nn.BatchNorm2d(MOVE_PLANES_NUM)
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.norm1(x))
        x = self.blocks(x)

        # policy head
        policy = self.policy_conv(x)
        policy = self.policy_bias(torch.flatten(policy, 1))

        # value head
        value = F.relu(self.value_norm1(self.value_conv1(x)))
        value = F.relu(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)

        return policy, value


def main():
    board = cshogi.Board()
    batch_size = 10
    torch_features = torch.empty(
        (FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=True
    )
    features = torch_features.numpy()
    set_board_features(board, features)
    print(features[14])


if __name__ == "__main__":
    main()
