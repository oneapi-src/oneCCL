#
# Copyright 2016-2020 Intel Corporation
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
#
#! /usr/bin/python3

import defusedxml.ElementTree as ET
import sys


def replace_tags(text):
    replace_with = [
        ("<", "&lt;"),
        (">", "&gt;"),
        ('"', "&quot;"),
        ("&lt;linebreak /&gt;", "<br />"),
        ("&lt;computeroutput&gt;", "`"),
        ("&lt;/computeroutput&gt;", "`"),
        ("&lt;itemizedlist&gt;", ""),
        ("&lt;/itemizedlist&gt;", ""),
        ("&lt;para&gt;", ""),
        ("&lt;/para&gt;", ""),
        ("&lt;listitem&gt;", " - "),
        ("&lt;/listitem&gt;", "\n")
    ]
    for replace_what, replace_with in replace_with:
        text = text.replace(replace_what, replace_with)
    return text


def item_to_string_without_tag(t):
    ret = [t.text if t.text else ""]
    ret += [ET.tostring(item).decode() for item in list(t)]
    return "".join(ret)


def parse_file(xml_file):
    tree = ET.parse(xml_file)
    ret = []
    for item in tree.iter():
        if item.tag == "compoundname":
            text = item_to_string_without_tag(item)
            ret.append(f"# {text}")
        elif item.tag == "definition":
            text = item_to_string_without_tag(item)
            text = text.replace("constexpr const char* ", "")
            ret.append(f"## {text}")
        elif item.tag in {
                "definition", "briefdescription", "detaileddescription"}:
            text = item_to_string_without_tag(item)
            ret.append(f"{text}")
        else:
            # this item is of no interest to us
            pass
    return ret


if __name__ == "__main__":
    ret = parse_file(sys.argv[1])
    print(replace_tags("\n\n".join(ret)))
