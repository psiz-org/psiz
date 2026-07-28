# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Initialization file for top-level package modules."""

from __future__ import annotations

from importlib import import_module


_LAZY_SUBMODULES = (
	"data",
	"keras",
	"migration",
	"mplot",
	"storage",
	"stochastic",
	"tfp",
	"utils",
)


def __getattr__(name):
	if name in _LAZY_SUBMODULES:
		module = import_module(f"psiz.{name}")
		globals()[name] = module
		return module
	raise AttributeError(f"module 'psiz' has no attribute '{name}'")


def __dir__():
	return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))
