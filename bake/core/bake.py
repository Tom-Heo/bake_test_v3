from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
from .heo import Heo as hh
from .palette import Palette as pp


class Bake:
    class OklabtoBakedColor(nn.Module):
        def __init__(self):
            super().__init__()
            """
            Normalized Oklab (Lp, ap, bp)
            Lp = 2*L - 1
            ap = 2*a
            bp = 2*b
            
            Macbeth ColorChecker 24 Reference Points (Float32 Precision)
            Lp, ap, bp 좌표계의 절대 기준점
            """
            MACBETH_REFS = [
                [-0.0597942, 0.0715682, 0.0689456],
                [0.4183730, 0.0844511, 0.0862138],
                [0.1485260, -0.0260145, -0.1197130],
                [0.0060500, -0.0867117, 0.1012940],
                [0.2456150, 0.0471529, -0.1390040],
                [0.4774960, -0.1772700, 0.0070673],
                [0.3507240, 0.1477280, 0.2429970],
                [-0.0020556, 0.0205867, -0.2380310],
                [0.1933670, 0.2535120, 0.0753901],
                [-0.1681210, 0.1244790, -0.1221760],
                [0.4962120, -0.1630230, 0.2584340],
                [0.5099230, 0.0561745, 0.2807750],
                [-0.1797940, 0.0282334, -0.2856320],
                [0.1974370, -0.2180970, 0.1560710],
                [0.0268793, 0.2907220, 0.1168620],
                [0.6649640, -0.0396422, 0.3292740],
                [0.1965930, 0.2874640, -0.0811799],
                [0.1415920, -0.1571350, -0.1312720],
                [0.9277470, -0.0007701, 0.0023781],
                [0.6655870, -0.0000168, -0.0001434],
                [0.4115140, -0.0000142, -0.0001216],
                [0.1583850, -0.0008645, 0.0027884],
                [-0.1009340, -0.0000091, -0.0000774],
                [-0.3499260, -0.0000066, -0.0000560],
            ]

            # 모델의 State로 등록 (Device 이동 자동화)
            self.register_buffer(
                "macbeth_refs",
                torch.tensor(MACBETH_REFS, dtype=torch.float32).view(1, 24, 3, 1, 1),
            )

            # 2. RBF Coefficients (Geometric Progression)
            self.coeffs = 0.5

        def forward(self, x):
            """
            Input:  x (B, 3, H, W) - Normalized Oklab (Lp, ap, bp)
            Lp = 2*L - 1
            ap = 2*a
            bp = 2*b
            Output: out (B, 30, H, W)
            """
            # --- Part 1: Signal Replication (6 Channels) ---
            # 원본 신호의 흐름을 강화하기 위해 2회 반복
            part1 = x.repeat(1, 2, 1, 1)

            # --- Part 2: RBF Global Context (24 Channels) ---
            # 1. Squared Euclidean Distance Calculation
            # (B, 1, 3, H, W) - (1, 24, 3, 1, 1) Broadcasting
            diff = x.unsqueeze(1) - self.macbeth_refs
            dist_sq = (diff**2).sum(dim=2)  # (B, 24, H, W)

            # 2. Gaussian RBF Application
            rbf_features = []

            # Gaussian Definition: exp( - d^2 / (2 * sigma^2) )
            gamma = 1.0 / (2.0 * (self.coeffs**2))
            rbf = torch.exp(-dist_sq * gamma)
            rbf_features.append(rbf)

            part2 = torch.cat(rbf_features, dim=1)

            # --- Final Concatenation (6 + 24 = 30 Channels) ---
            baked_color = torch.cat([part1, part2], dim=1)

            return baked_color

    class BakedColortoOklab(nn.Module):
        """
        30채널 BakedColor에서 순수 신호(Lp, ap, bp)를 추출하여 평균을 냄.
        Bake.OklabtoBakedColor의 역연산 개념.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 30, H, W)

            # Part 1: Signal Replication 영역 (0~5번 채널) 추출
            # 구조: [L, a, b, L, a, b, ...] 형태가 2번 반복됨
            signal_part = x[:, :6, :, :]

            # (B, 2, 3, H, W)로 Reshape하여 2번의 반복을 차원으로 분리
            # dim 1이 반복 횟수(2), dim 2가 채널(3: L,a,b)
            reshaped = signal_part.view(x.size(0), 2, 3, x.size(2), x.size(3))

            # 2개의 반복된 신호에 대해 평균을 구함 -> 노이즈 캔슬링 효과
            avg_signal = reshaped.mean(dim=1)  # (B, 3, H, W)

            return avg_signal

    class BakedColortoColorEmbedding(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = nn.Conv2d(30, 30, 1, 1)
            self.conv2 = nn.Conv2d(30, 30, 1, 1)
            self.conv3 = nn.Conv2d(30, 30, 1, 1)
            self.act1 = hh.HeLU2d(30)
            self.act2 = hh.HeLU2d(30)
            self.act3 = hh.HeLU2d(30)
            self.gate1 = hh.HeoGate2d(30)
            self.gate2 = hh.HeoGate2d(30)
            self.gate3 = hh.HeoGate2d(30)

        def forward(self, baked_color):
            """
            Input: baked_color (B, 30, H, W)
            """

            x1 = self.conv1(baked_color)
            x1 = self.act1(x1)
            x1 = self.gate1(baked_color, x1)

            x2 = self.conv2(x1)
            x2 = self.act2(x2)
            x2 = self.gate2(x1, x2)

            x3 = self.conv3(x2)
            x3 = self.act3(x3)
            color_embedding = self.gate3(x2, x3)

            return color_embedding

    class BakeLoss(nn.Module):
        def __init__(self, epsilon=0.001):
            super().__init__()
            self.epsilon = epsilon
            self.epsilon_char = 1e-8

        def forward(self, pred, target):
            pred = pred.float()  # FP32 강제
            target = target.float()

            diff = pred - target
            charbonnier = torch.sqrt(diff**2 + self.epsilon_char**2)

            loss = torch.log(1 + charbonnier / self.epsilon)

            return loss.mean()

    class BakeNet(nn.Module):
        def __init__(self, dim=30):
            super().__init__()

            self.nemo33_1 = hh.NeMO33(dim)
            self.nemo55_2 = hh.NeMO55(dim)
            self.nemo77_3 = hh.NeMO77(dim)
            self.nemo99_4 = hh.NeMO99(dim)
            self.nemo1111_5 = hh.NeMO1111(dim)
            self.nemo1313_6 = hh.NeMO1313(dim)
            self.nemo1515_7 = hh.NeMO1515(dim)
            self.nemo1717_8 = hh.NeMO1717(dim)
            self.nemo1919_9 = hh.NeMO1919(dim)
            self.nemo2121_10 = hh.NeMO2121(dim)
            self.nemo1919_11 = hh.NeMO1919(dim)
            self.nemo1717_12 = hh.NeMO1717(dim)
            self.nemo1515_13 = hh.NeMO1515(dim)
            self.nemo1313_14 = hh.NeMO1313(dim)
            self.nemo1111_15 = hh.NeMO1111(dim)
            self.nemo99_16 = hh.NeMO99(dim)
            self.nemo77_17 = hh.NeMO77(dim)
            self.nemo55_18 = hh.NeMO55(dim)
            self.nemo33_19 = hh.NeMO33(dim)

            self.color_embedding_1 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_2 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_3 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_4 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_5 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_6 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_7 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_8 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_9 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_10 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_11 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_12 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_13 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_14 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_15 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_16 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_17 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_18 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_19 = Bake.BakedColortoColorEmbedding()

            self.color_embedding_gate_1 = hh.HeoGate2d(dim)
            self.color_embedding_gate_2 = hh.HeoGate2d(dim)
            self.color_embedding_gate_3 = hh.HeoGate2d(dim)
            self.color_embedding_gate_4 = hh.HeoGate2d(dim)
            self.color_embedding_gate_5 = hh.HeoGate2d(dim)
            self.color_embedding_gate_6 = hh.HeoGate2d(dim)
            self.color_embedding_gate_7 = hh.HeoGate2d(dim)
            self.color_embedding_gate_8 = hh.HeoGate2d(dim)
            self.color_embedding_gate_9 = hh.HeoGate2d(dim)
            self.color_embedding_gate_10 = hh.HeoGate2d(dim)
            self.color_embedding_gate_11 = hh.HeoGate2d(dim)
            self.color_embedding_gate_12 = hh.HeoGate2d(dim)
            self.color_embedding_gate_13 = hh.HeoGate2d(dim)
            self.color_embedding_gate_14 = hh.HeoGate2d(dim)
            self.color_embedding_gate_15 = hh.HeoGate2d(dim)
            self.color_embedding_gate_16 = hh.HeoGate2d(dim)
            self.color_embedding_gate_17 = hh.HeoGate2d(dim)
            self.color_embedding_gate_18 = hh.HeoGate2d(dim)
            self.color_embedding_gate_19 = hh.HeoGate2d(dim)

            self.residual_gate_1 = hh.HeoGate2d(dim)
            self.residual_gate_2 = hh.HeoGate2d(dim)
            self.residual_gate_3 = hh.HeoGate2d(dim)
            self.residual_gate_4 = hh.HeoGate2d(dim)
            self.residual_gate_5 = hh.HeoGate2d(dim)
            self.residual_gate_6 = hh.HeoGate2d(dim)
            self.residual_gate_7 = hh.HeoGate2d(dim)
            self.residual_gate_8 = hh.HeoGate2d(dim)
            self.residual_gate_9 = hh.HeoGate2d(dim)
            self.residual_gate_10 = hh.HeoGate2d(dim)
            self.residual_gate_11 = hh.HeoGate2d(dim)
            self.residual_gate_12 = hh.HeoGate2d(dim)
            self.residual_gate_13 = hh.HeoGate2d(dim)
            self.residual_gate_14 = hh.HeoGate2d(dim)
            self.residual_gate_15 = hh.HeoGate2d(dim)
            self.residual_gate_16 = hh.HeoGate2d(dim)
            self.residual_gate_17 = hh.HeoGate2d(dim)
            self.residual_gate_18 = hh.HeoGate2d(dim)
            self.residual_gate_19 = hh.HeoGate2d(dim)

        def forward(self, baked_x_lr):

            color_embedding_1 = self.color_embedding_1(baked_x_lr)
            color_embedding_2 = self.color_embedding_2(baked_x_lr)
            color_embedding_3 = self.color_embedding_3(baked_x_lr)
            color_embedding_4 = self.color_embedding_4(baked_x_lr)
            color_embedding_5 = self.color_embedding_5(baked_x_lr)
            color_embedding_6 = self.color_embedding_6(baked_x_lr)
            color_embedding_7 = self.color_embedding_7(baked_x_lr)
            color_embedding_8 = self.color_embedding_8(baked_x_lr)
            color_embedding_9 = self.color_embedding_9(baked_x_lr)
            color_embedding_10 = self.color_embedding_10(baked_x_lr)
            color_embedding_11 = self.color_embedding_11(baked_x_lr)
            color_embedding_12 = self.color_embedding_12(baked_x_lr)
            color_embedding_13 = self.color_embedding_13(baked_x_lr)
            color_embedding_14 = self.color_embedding_14(baked_x_lr)
            color_embedding_15 = self.color_embedding_15(baked_x_lr)
            color_embedding_16 = self.color_embedding_16(baked_x_lr)
            color_embedding_17 = self.color_embedding_17(baked_x_lr)
            color_embedding_18 = self.color_embedding_18(baked_x_lr)
            color_embedding_19 = self.color_embedding_19(baked_x_lr)

            x1 = self.color_embedding_gate_1(baked_x_lr, color_embedding_1)
            x2 = self.nemo33_1(x1)
            x2 = self.residual_gate_1(x2, x1)

            x2 = self.color_embedding_gate_2(x2, color_embedding_2)
            x3 = self.nemo55_2(x2)
            x3 = self.residual_gate_2(x3, x2)

            x3 = self.color_embedding_gate_3(x3, color_embedding_3)
            x4 = self.nemo77_3(x3)
            x4 = self.residual_gate_3(x4, x3)

            x4 = self.color_embedding_gate_4(x4, color_embedding_4)
            x5 = self.nemo99_4(x4)
            x5 = self.residual_gate_4(x5, x4)

            x5 = self.color_embedding_gate_5(x5, color_embedding_5)
            x6 = self.nemo1111_5(x5)
            x6 = self.residual_gate_5(x6, x5)

            x6 = self.color_embedding_gate_6(x6, color_embedding_6)
            x7 = self.nemo1313_6(x6)
            x7 = self.residual_gate_6(x7, x6)

            x7 = self.color_embedding_gate_7(x7, color_embedding_7)
            x8 = self.nemo1515_7(x7)
            x8 = self.residual_gate_7(x8, x7)

            x8 = self.color_embedding_gate_8(x8, color_embedding_8)
            x9 = self.nemo1717_8(x8)
            x9 = self.residual_gate_8(x9, x8)

            x9 = self.color_embedding_gate_9(x9, color_embedding_9)
            x10 = self.nemo1919_9(x9)
            x10 = self.residual_gate_9(x10, x9)

            x10 = self.color_embedding_gate_10(x10, color_embedding_10)
            x11 = self.nemo2121_10(x10)
            x11 = self.residual_gate_10(x11, x10)

            x11 = self.color_embedding_gate_11(x11, color_embedding_11)
            x12 = self.nemo1919_11(x11)
            x12 = self.residual_gate_11(x12, x11)

            x12 = self.color_embedding_gate_12(x12, color_embedding_12)
            x13 = self.nemo1717_12(x12)
            x13 = self.residual_gate_12(x13, x12)

            x13 = self.color_embedding_gate_13(x13, color_embedding_13)
            x14 = self.nemo1515_13(x13)
            x14 = self.residual_gate_13(x14, x13)

            x14 = self.color_embedding_gate_14(x14, color_embedding_14)
            x15 = self.nemo1313_14(x14)
            x15 = self.residual_gate_14(x15, x14)

            x15 = self.color_embedding_gate_15(x15, color_embedding_15)
            x16 = self.nemo1111_15(x15)
            x16 = self.residual_gate_15(x16, x15)

            x16 = self.color_embedding_gate_16(x16, color_embedding_16)
            x17 = self.nemo99_16(x16)
            x17 = self.residual_gate_16(x17, x16)

            x17 = self.color_embedding_gate_17(x17, color_embedding_17)
            x18 = self.nemo77_17(x17)
            x18 = self.residual_gate_17(x18, x17)

            x18 = self.color_embedding_gate_18(x18, color_embedding_18)
            x19 = self.nemo55_18(x18)
            x19 = self.residual_gate_18(x19, x18)

            x19 = self.color_embedding_gate_19(x19, color_embedding_19)
            x20 = self.nemo33_19(x19)
            x20 = self.residual_gate_19(x20, x19)

            return x20
