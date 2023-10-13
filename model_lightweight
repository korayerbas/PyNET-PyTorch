# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import torch.nn as nn
import torch

class PyNET(nn.Module):

    def __init__(self, level, instance_norm=True, instance_norm_level_1=False):
        super(PyNET, self).__init__()

        self.level = level

        self.conv_l1_d1 = ConvMultiBlock(4, 32, 3, instance_norm=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_l2_d1 = ConvMultiBlock(32, 64, 3, instance_norm=instance_norm)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv_l3_d1 = ConvMultiBlock(64, 128, 3, instance_norm=instance_norm)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # -------------------------------------

        self.conv_l4_d1 = ConvMultiBlock(128, 256, 3, instance_norm=instance_norm)
        self.conv_l4_d2 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)
        self.conv_l4_d3 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)
        self.conv_l4_d4 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)

        self.conv_t3b = UpsampleConvLayer(256, 128, 3)

        self.conv_l4_out = ConvLayer(256, 3, kernel_size=3, stride=1, relu=False)
        self.output_l4 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l3_d3 = ConvMultiBlock(256, 128, 3, instance_norm=instance_norm)

        self.conv_t2b = UpsampleConvLayer(128, 64, 3)

        self.conv_l3_out = ConvLayer(128, 3, kernel_size=3, stride=1, relu=False)
        self.output_l3 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l2_d3 = ConvMultiBlock(128, 64, 3, instance_norm=instance_norm)
        
        self.conv_t1b = UpsampleConvLayer(64, 32, 3)
        self.conv_l2_out = ConvLayer(64, 3, kernel_size=3, stride=1, relu=False)
        self.output_l2 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l1_d3 = ConvMultiBlock(64, 32, 3, instance_norm=instance_norm)
        
        self.conv_t0b = UpsampleConvLayer(32, 16, 3)
        self.conv_l1_out = ConvLayer(32, 3, kernel_size=3, stride=1, relu=False)
        self.output_l1 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l0_d1 = ConvLayer(16, 3, kernel_size=3, stride=1, relu=False)
        self.output_l0 = nn.Sigmoid()

    def level_4(self, pool3):
        
        conv_l4_d1 = self.conv_l4_d1(pool3)
        conv_l4_d2 = self.conv_l4_d2(conv_l4_d1)
        conv_l4_d3 = self.conv_l4_d3(conv_l4_d2)
        conv_l4_d4 = self.conv_l4_d4(conv_l4_d3)
        conv_t3b = self.conv_t3b(conv_l4_d4)
        conv_l4_out = self.conv_l4_out(conv_l4_d4)
        output_l4 = self.output_l4(conv_l4_out)

        return output_l4, conv_t3b

    def level_3(self, conv_l3_d1, conv_t3b):

        conv_l3_d2 = torch.cat([conv_l3_d1, conv_t3b], 1)
        conv_l3_d3 = self.conv_l3_d3(conv_l3_d2)
        conv_t2b = self.conv_t2b(conv_l3_d3)
        conv_l3_out = self.conv_l3_out(conv_l3_d3)
        output_l3 = self.output_l3(conv_l3_out)
        
        return output_l3, conv_t2b

    def level_2(self, conv_l2_d1, conv_t2b):

        conv_l2_d2 = torch.cat([conv_l2_d1, conv_t2b], 1)
        conv_l2_d3 = self.conv_l2_d3(conv_l2_d2)
        conv_t1b = self.conv_t1b(conv_l2_d3)
        conv_l2_out = self.conv_l2_out(conv_l2_d3)
        output_l2 = self.output_l2(conv_l2_out)

        return output_l2, conv_t1b

    def level_1(self, conv_l1_d1, conv_t1b):

        conv_l1_d2 = torch.cat([conv_l1_d1, conv_t1b], 1)
        conv_l1_d3 = self.conv_l1_d3(conv_l1_d2)
        conv_t0b = self.conv_t0b(conv_l1_d3)
        conv_l1_out = self.conv_l1_out(conv_l1_d3)
        output_l1 = self.output_l1(conv_l1_out)

        return output_l1, conv_t0b

    def level_0(self, conv_t0b):

        conv_l0_d1 = self.conv_l0_d1(conv_t0b)
        output_l0 = self.output_l0(conv_l0_d1)

        return output_l0

    def forward(self, x):

        conv_l1_d1 = self.conv_l1_d1(x)
        pool1 = self.pool1(conv_l1_d1)

        conv_l2_d1 = self.conv_l2_d1(pool1)
        pool2 = self.pool2(conv_l2_d1)

        conv_l3_d1 = self.conv_l3_d1(pool2)
        pool3 = self.pool3(conv_l3_d1)

        output_l4, conv_t3b =self.level_4(pool3)

        if self.level < 4:
            output_l3, conv_t2b = self.level_3(conv_l3_d1, conv_t3b)
        if self.level < 3:
            output_l2, conv_t1b = self.level_2(conv_l2_d1, conv_t2b)
        if self.level < 2:
            output_l1, conv_t0b = self.level_1(conv_l1_d1, conv_t1b)
        if self.level < 1:
            output_l0 = self.level_0(conv_t0b)

        if self.level == 0:
            enhanced = output_l0
        if self.level == 1:
            enhanced = output_l1
        if self.level == 2:
            enhanced = output_l2
        if self.level == 3:
            enhanced = output_l3
        if self.level == 4:
            enhanced = output_l4

        return enhanced


class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, max_conv_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        self.max_conv_size = max_conv_size

        self.conv_3a = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)
        self.conv_3b = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 5:
            self.conv_5a = ConvLayer(in_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)
            self.conv_5b = ConvLayer(out_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 7:
            self.conv_7a = ConvLayer(in_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)
            self.conv_7b = ConvLayer(out_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 9:
            self.conv_9a = ConvLayer(in_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)
            self.conv_9b = ConvLayer(out_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)

    def forward(self, x):

        out_3 = self.conv_3a(x)
        output_tensor = self.conv_3b(out_3)

        if self.max_conv_size >= 5:
            out_5 = self.conv_5a(x)
            out_5 = self.conv_5b(out_5)
            output_tensor = torch.cat([output_tensor, out_5], 1)

        if self.max_conv_size >= 7:
            out_7 = self.conv_7a(x)
            out_7 = self.conv_7b(out_7)
            output_tensor = torch.cat([output_tensor, out_7], 1)

        if self.max_conv_size >= 9:
            out_9 = self.conv_9a(x)
            out_9 = self.conv_9b(out_9)
            output_tensor = torch.cat([output_tensor, out_9], 1)

        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, stride=1, relu=True):

        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)

        return out
