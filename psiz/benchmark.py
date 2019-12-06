# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
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
# ==============================================================================

"""Module for benchmarking."""

import datetime
import platform
# import socket
import re
import uuid

import psutil


def system_info():
    """Get system information."""
    sys_info = {}
    sys_info['platform'] = platform.system()
    sys_info['platform-release'] = platform.release()
    sys_info['platform-version'] = platform.version()
    sys_info['architecture'] = platform.machine()
    # sys_info['hostname'] = socket.gethostname()
    # sys_info['ip-address'] = socket.gethostbyname(socket.gethostname())
    sys_info['mac-address'] = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
    sys_info['processor'] = platform.processor()
    sys_info['ram'] = str(
        round(psutil.virtual_memory().total / (1024.0 **3))
    ) + " GB"
    return sys_info


def benchmark_filename():
    """Generate a default filename for a benchmark report."""
    d_str = str(datetime.datetime.now())
    d_str = d_str[0:-7]
    s = 'mark_{0}.json'.format(d_str)
    s = s.replace(' ', '_')
    return s
