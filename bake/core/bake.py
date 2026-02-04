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

            self.nemo1 = hh.NeMO33(dim)
            self.nemo2 = hh.NeMO33(dim)
            self.nemo3 = hh.NeMO33(dim)
            self.nemo4 = hh.NeMO33(dim)
            self.nemo5 = hh.NeMO33(dim)
            self.nemo6 = hh.NeMO33(dim)
            self.nemo7 = hh.NeMO33(dim)
            self.nemo8 = hh.NeMO33(dim)
            self.nemo9 = hh.NeMO33(dim)
            self.nemo10 = hh.NeMO33(dim)
            self.nemo11 = hh.NeMO33(dim)
            self.nemo12 = hh.NeMO33(dim)
            self.nemo13 = hh.NeMO33(dim)
            self.nemo14 = hh.NeMO33(dim)
            self.nemo15 = hh.NeMO33(dim)
            self.nemo16 = hh.NeMO33(dim)
            self.nemo17 = hh.NeMO33(dim)
            self.nemo18 = hh.NeMO33(dim)
            self.nemo19 = hh.NeMO33(dim)
            self.nemo20 = hh.NeMO33(dim)
            self.nemo21 = hh.NeMO33(dim)
            self.nemo22 = hh.NeMO33(dim)
            self.nemo23 = hh.NeMO33(dim)
            self.nemo24 = hh.NeMO33(dim)
            self.nemo25 = hh.NeMO33(dim)
            self.nemo26 = hh.NeMO33(dim)
            self.nemo27 = hh.NeMO33(dim)
            self.nemo28 = hh.NeMO33(dim)
            self.nemo29 = hh.NeMO33(dim)
            self.nemo30 = hh.NeMO33(dim)
            self.nemo31 = hh.NeMO33(dim)
            self.nemo32 = hh.NeMO33(dim)
            self.nemo33 = hh.NeMO33(dim)
            self.nemo34 = hh.NeMO33(dim)
            self.nemo35 = hh.NeMO33(dim)
            self.nemo36 = hh.NeMO33(dim)
            self.nemo37 = hh.NeMO33(dim)
            self.nemo38 = hh.NeMO33(dim)
            self.nemo39 = hh.NeMO33(dim)
            self.nemo40 = hh.NeMO33(dim)
            self.nemo41 = hh.NeMO33(dim)
            self.nemo42 = hh.NeMO33(dim)
            self.nemo43 = hh.NeMO33(dim)
            self.nemo44 = hh.NeMO33(dim)
            self.nemo45 = hh.NeMO33(dim)
            self.nemo46 = hh.NeMO33(dim)
            self.nemo47 = hh.NeMO33(dim)
            self.nemo48 = hh.NeMO33(dim)
            self.nemo49 = hh.NeMO33(dim)
            self.nemo50 = hh.NeMO33(dim)

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
            self.color_embedding_20 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_21 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_22 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_23 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_24 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_25 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_26 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_27 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_28 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_29 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_30 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_31 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_32 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_33 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_34 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_35 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_36 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_37 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_38 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_39 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_40 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_41 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_42 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_43 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_44 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_45 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_46 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_47 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_48 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_49 = Bake.BakedColortoColorEmbedding()
            self.color_embedding_50 = Bake.BakedColortoColorEmbedding()

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
            self.color_embedding_gate_20 = hh.HeoGate2d(dim)
            self.color_embedding_gate_21 = hh.HeoGate2d(dim)
            self.color_embedding_gate_22 = hh.HeoGate2d(dim)
            self.color_embedding_gate_23 = hh.HeoGate2d(dim)
            self.color_embedding_gate_24 = hh.HeoGate2d(dim)
            self.color_embedding_gate_25 = hh.HeoGate2d(dim)
            self.color_embedding_gate_26 = hh.HeoGate2d(dim)
            self.color_embedding_gate_27 = hh.HeoGate2d(dim)
            self.color_embedding_gate_28 = hh.HeoGate2d(dim)
            self.color_embedding_gate_29 = hh.HeoGate2d(dim)
            self.color_embedding_gate_30 = hh.HeoGate2d(dim)
            self.color_embedding_gate_31 = hh.HeoGate2d(dim)
            self.color_embedding_gate_32 = hh.HeoGate2d(dim)
            self.color_embedding_gate_33 = hh.HeoGate2d(dim)
            self.color_embedding_gate_34 = hh.HeoGate2d(dim)
            self.color_embedding_gate_35 = hh.HeoGate2d(dim)
            self.color_embedding_gate_36 = hh.HeoGate2d(dim)
            self.color_embedding_gate_37 = hh.HeoGate2d(dim)
            self.color_embedding_gate_38 = hh.HeoGate2d(dim)
            self.color_embedding_gate_39 = hh.HeoGate2d(dim)
            self.color_embedding_gate_40 = hh.HeoGate2d(dim)
            self.color_embedding_gate_41 = hh.HeoGate2d(dim)
            self.color_embedding_gate_42 = hh.HeoGate2d(dim)
            self.color_embedding_gate_43 = hh.HeoGate2d(dim)
            self.color_embedding_gate_44 = hh.HeoGate2d(dim)
            self.color_embedding_gate_45 = hh.HeoGate2d(dim)
            self.color_embedding_gate_46 = hh.HeoGate2d(dim)
            self.color_embedding_gate_47 = hh.HeoGate2d(dim)
            self.color_embedding_gate_48 = hh.HeoGate2d(dim)
            self.color_embedding_gate_49 = hh.HeoGate2d(dim)
            self.color_embedding_gate_50 = hh.HeoGate2d(dim)

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
            self.residual_gate_20 = hh.HeoGate2d(dim)
            self.residual_gate_21 = hh.HeoGate2d(dim)
            self.residual_gate_22 = hh.HeoGate2d(dim)
            self.residual_gate_23 = hh.HeoGate2d(dim)
            self.residual_gate_24 = hh.HeoGate2d(dim)
            self.residual_gate_25 = hh.HeoGate2d(dim)
            self.residual_gate_26 = hh.HeoGate2d(dim)
            self.residual_gate_27 = hh.HeoGate2d(dim)
            self.residual_gate_28 = hh.HeoGate2d(dim)
            self.residual_gate_29 = hh.HeoGate2d(dim)
            self.residual_gate_30 = hh.HeoGate2d(dim)
            self.residual_gate_31 = hh.HeoGate2d(dim)
            self.residual_gate_32 = hh.HeoGate2d(dim)
            self.residual_gate_33 = hh.HeoGate2d(dim)
            self.residual_gate_34 = hh.HeoGate2d(dim)
            self.residual_gate_35 = hh.HeoGate2d(dim)
            self.residual_gate_36 = hh.HeoGate2d(dim)
            self.residual_gate_37 = hh.HeoGate2d(dim)
            self.residual_gate_38 = hh.HeoGate2d(dim)
            self.residual_gate_39 = hh.HeoGate2d(dim)
            self.residual_gate_40 = hh.HeoGate2d(dim)
            self.residual_gate_41 = hh.HeoGate2d(dim)
            self.residual_gate_42 = hh.HeoGate2d(dim)
            self.residual_gate_43 = hh.HeoGate2d(dim)
            self.residual_gate_44 = hh.HeoGate2d(dim)
            self.residual_gate_45 = hh.HeoGate2d(dim)
            self.residual_gate_46 = hh.HeoGate2d(dim)
            self.residual_gate_47 = hh.HeoGate2d(dim)
            self.residual_gate_48 = hh.HeoGate2d(dim)
            self.residual_gate_49 = hh.HeoGate2d(dim)
            self.residual_gate_50 = hh.HeoGate2d(dim)

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
            color_embedding_20 = self.color_embedding_20(baked_x_lr)
            color_embedding_21 = self.color_embedding_21(baked_x_lr)
            color_embedding_22 = self.color_embedding_22(baked_x_lr)
            color_embedding_23 = self.color_embedding_23(baked_x_lr)
            color_embedding_24 = self.color_embedding_24(baked_x_lr)
            color_embedding_25 = self.color_embedding_25(baked_x_lr)
            color_embedding_26 = self.color_embedding_26(baked_x_lr)
            color_embedding_27 = self.color_embedding_27(baked_x_lr)
            color_embedding_28 = self.color_embedding_28(baked_x_lr)
            color_embedding_29 = self.color_embedding_29(baked_x_lr)
            color_embedding_30 = self.color_embedding_30(baked_x_lr)
            color_embedding_31 = self.color_embedding_31(baked_x_lr)
            color_embedding_32 = self.color_embedding_32(baked_x_lr)
            color_embedding_33 = self.color_embedding_33(baked_x_lr)
            color_embedding_34 = self.color_embedding_34(baked_x_lr)
            color_embedding_35 = self.color_embedding_35(baked_x_lr)
            color_embedding_36 = self.color_embedding_36(baked_x_lr)
            color_embedding_37 = self.color_embedding_37(baked_x_lr)
            color_embedding_38 = self.color_embedding_38(baked_x_lr)
            color_embedding_39 = self.color_embedding_39(baked_x_lr)
            color_embedding_40 = self.color_embedding_40(baked_x_lr)
            color_embedding_41 = self.color_embedding_41(baked_x_lr)
            color_embedding_42 = self.color_embedding_42(baked_x_lr)
            color_embedding_43 = self.color_embedding_43(baked_x_lr)
            color_embedding_44 = self.color_embedding_44(baked_x_lr)
            color_embedding_45 = self.color_embedding_45(baked_x_lr)
            color_embedding_46 = self.color_embedding_46(baked_x_lr)
            color_embedding_47 = self.color_embedding_47(baked_x_lr)
            color_embedding_48 = self.color_embedding_48(baked_x_lr)
            color_embedding_49 = self.color_embedding_49(baked_x_lr)
            color_embedding_50 = self.color_embedding_50(baked_x_lr)

            x1 = self.color_embedding_gate_1(baked_x_lr, color_embedding_1)
            x2 = self.nemo1(x1)
            x2 = self.residual_gate_1(x2, x1)

            x2 = self.color_embedding_gate_2(x2, color_embedding_2)
            x3 = self.nemo2(x2)
            x3 = self.residual_gate_2(x3, x2)

            x3 = self.color_embedding_gate_3(x3, color_embedding_3)
            x4 = self.nemo3(x3)
            x4 = self.residual_gate_3(x4, x3)

            x4 = self.color_embedding_gate_4(x4, color_embedding_4)
            x5 = self.nemo4(x4)
            x5 = self.residual_gate_4(x5, x4)

            x5 = self.color_embedding_gate_5(x5, color_embedding_5)
            x6 = self.nemo5(x5)
            x6 = self.residual_gate_5(x6, x5)

            x6 = self.color_embedding_gate_6(x6, color_embedding_6)
            x7 = self.nemo6(x6)
            x7 = self.residual_gate_6(x7, x6)

            x7 = self.color_embedding_gate_7(x7, color_embedding_7)
            x8 = self.nemo7(x7)
            x8 = self.residual_gate_7(x8, x7)

            x8 = self.color_embedding_gate_8(x8, color_embedding_8)
            x9 = self.nemo8(x8)
            x9 = self.residual_gate_8(x9, x8)

            x9 = self.color_embedding_gate_9(x9, color_embedding_9)
            x10 = self.nemo9(x9)
            x10 = self.residual_gate_9(x10, x9)

            x10 = self.color_embedding_gate_10(x10, color_embedding_10)
            x11 = self.nemo10(x10)
            x11 = self.residual_gate_10(x11, x10)

            x11 = self.color_embedding_gate_11(x11, color_embedding_11)
            x12 = self.nemo11(x11)
            x12 = self.residual_gate_11(x12, x11)

            x12 = self.color_embedding_gate_12(x12, color_embedding_12)
            x13 = self.nemo12(x12)
            x13 = self.residual_gate_12(x13, x12)

            x13 = self.color_embedding_gate_13(x13, color_embedding_13)
            x14 = self.nemo13(x13)
            x14 = self.residual_gate_13(x14, x13)

            x14 = self.color_embedding_gate_14(x14, color_embedding_14)
            x15 = self.nemo14(x14)
            x15 = self.residual_gate_14(x15, x14)

            x15 = self.color_embedding_gate_15(x15, color_embedding_15)
            x16 = self.nemo15(x15)
            x16 = self.residual_gate_15(x16, x15)

            x16 = self.color_embedding_gate_16(x16, color_embedding_16)
            x17 = self.nemo16(x16)
            x17 = self.residual_gate_16(x17, x16)

            x17 = self.color_embedding_gate_17(x17, color_embedding_17)
            x18 = self.nemo17(x17)
            x18 = self.residual_gate_17(x18, x17)

            x18 = self.color_embedding_gate_18(x18, color_embedding_18)
            x19 = self.nemo18(x18)
            x19 = self.residual_gate_18(x19, x18)

            x19 = self.color_embedding_gate_19(x19, color_embedding_19)
            x20 = self.nemo19(x19)
            x20 = self.residual_gate_19(x20, x19)

            x20 = self.color_embedding_gate_20(x20, color_embedding_20)
            x21 = self.nemo20(x20)
            x21 = self.residual_gate_20(x21, x20)

            x21 = self.color_embedding_gate_21(x21, color_embedding_21)
            x22 = self.nemo21(x21)
            x22 = self.residual_gate_21(x22, x21)

            x22 = self.color_embedding_gate_22(x22, color_embedding_22)
            x23 = self.nemo22(x22)
            x23 = self.residual_gate_22(x23, x22)

            x23 = self.color_embedding_gate_23(x23, color_embedding_23)
            x24 = self.nemo23(x23)
            x24 = self.residual_gate_23(x24, x23)

            x24 = self.color_embedding_gate_24(x24, color_embedding_24)
            x25 = self.nemo24(x24)
            x25 = self.residual_gate_24(x25, x24)

            x25 = self.color_embedding_gate_25(x25, color_embedding_25)
            x26 = self.nemo25(x25)
            x26 = self.residual_gate_25(x26, x25)

            x26 = self.color_embedding_gate_26(x26, color_embedding_26)
            x27 = self.nemo26(x26)
            x27 = self.residual_gate_26(x27, x26)

            x27 = self.color_embedding_gate_27(x27, color_embedding_27)
            x28 = self.nemo27(x27)
            x28 = self.residual_gate_27(x28, x27)

            x28 = self.color_embedding_gate_28(x28, color_embedding_28)
            x29 = self.nemo28(x28)
            x29 = self.residual_gate_28(x29, x28)

            x29 = self.color_embedding_gate_29(x29, color_embedding_29)
            x30 = self.nemo29(x29)
            x30 = self.residual_gate_29(x30, x29)

            x30 = self.color_embedding_gate_30(x30, color_embedding_30)
            x31 = self.nemo30(x30)
            x31 = self.residual_gate_30(x31, x30)

            x31 = self.color_embedding_gate_31(x31, color_embedding_31)
            x32 = self.nemo31(x31)
            x32 = self.residual_gate_31(x32, x31)

            x32 = self.color_embedding_gate_32(x32, color_embedding_32)
            x33 = self.nemo32(x32)
            x33 = self.residual_gate_32(x33, x32)

            x33 = self.color_embedding_gate_33(x33, color_embedding_33)
            x34 = self.nemo33(x33)
            x34 = self.residual_gate_33(x34, x33)

            x34 = self.color_embedding_gate_34(x34, color_embedding_34)
            x35 = self.nemo34(x34)
            x35 = self.residual_gate_34(x35, x34)

            x35 = self.color_embedding_gate_35(x35, color_embedding_35)
            x36 = self.nemo35(x35)
            x36 = self.residual_gate_35(x36, x35)

            x36 = self.color_embedding_gate_36(x36, color_embedding_36)
            x37 = self.nemo36(x36)
            x37 = self.residual_gate_36(x37, x36)

            x37 = self.color_embedding_gate_37(x37, color_embedding_37)
            x38 = self.nemo37(x37)
            x38 = self.residual_gate_37(x38, x37)

            x38 = self.color_embedding_gate_38(x38, color_embedding_38)
            x39 = self.nemo38(x38)
            x39 = self.residual_gate_38(x39, x38)

            x39 = self.color_embedding_gate_39(x39, color_embedding_39)
            x40 = self.nemo39(x39)
            x40 = self.residual_gate_39(x40, x39)

            x40 = self.color_embedding_gate_40(x40, color_embedding_40)
            x41 = self.nemo40(x40)
            x41 = self.residual_gate_40(x41, x40)

            x41 = self.color_embedding_gate_41(x41, color_embedding_41)
            x42 = self.nemo41(x41)
            x42 = self.residual_gate_41(x42, x41)

            x42 = self.color_embedding_gate_42(x42, color_embedding_42)
            x43 = self.nemo42(x42)
            x43 = self.residual_gate_42(x43, x42)

            x43 = self.color_embedding_gate_43(x43, color_embedding_43)
            x44 = self.nemo43(x43)
            x44 = self.residual_gate_43(x44, x43)

            x44 = self.color_embedding_gate_44(x44, color_embedding_44)
            x45 = self.nemo44(x44)
            x45 = self.residual_gate_44(x45, x44)

            x45 = self.color_embedding_gate_45(x45, color_embedding_45)
            x46 = self.nemo45(x45)
            x46 = self.residual_gate_45(x46, x45)

            x46 = self.color_embedding_gate_46(x46, color_embedding_46)
            x47 = self.nemo46(x46)
            x47 = self.residual_gate_46(x47, x46)

            x47 = self.color_embedding_gate_47(x47, color_embedding_47)
            x48 = self.nemo47(x47)
            x48 = self.residual_gate_47(x48, x47)

            x48 = self.color_embedding_gate_48(x48, color_embedding_48)
            x49 = self.nemo48(x48)
            x49 = self.residual_gate_48(x49, x48)

            x49 = self.color_embedding_gate_49(x49, color_embedding_49)
            x50 = self.nemo49(x49)
            x50 = self.residual_gate_49(x50, x49)

            x50 = self.color_embedding_gate_50(x50, color_embedding_50)
            x51 = self.nemo50(x50)
            x51 = self.residual_gate_50(x51, x50)

            return x51
