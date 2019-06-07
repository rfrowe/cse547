#!/usr/bin/env python3
"""
cmd_line.py
---

CMD Line parsing utilities.

Taken from:

https://github.com/joseph-zhong/VideoSummarization/blob/master/src/utils/cmd_line.py

"""
import argparse
import inspect
import re
import sys
from types import GeneratorType

from typing import Union, Optional, IO, List, Tuple

try:
    from typing import GenericMeta  # python 3.6
except ImportError:
    # in 3.7, genericmeta doesn't exist but we don't need it
    class GenericMeta(type): pass

import utils.utility as _util


class ArgParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser which injects function doc comment in help print.
    """
    _HELP = None

    def print_help(self, file: Optional[IO[str]] = None) -> None:
        if file is None:
            file = sys.stdout
        if self._HELP:
            self._print_message(self._HELP, file)
            self._print_message("\n\n", file)

        super().print_help(file)

    @classmethod
    def set_help(cls, help):
        cls._HELP = help


def add_boolean_argument(parser, name, help=None, default=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--" + name,
        nargs="?",
        default=default,
        const=True,
        type=_util.str_to_bool,
        help=help)
    group.add_argument(("--no_" if default else "--yes_") + name,
                       dest=name,
                       action="store_" + ("false" if default else "true"),
                       help="(anti) {}".format(help) if help is not None else None)


def _parse_docstr(docstr: str):
    if ":param" in docstr:
        start = docstr.index(":param")
        description, rest = docstr[0:start].strip(), docstr[start:].strip()

        params = {}
        while rest.startswith(":param"):
            rest = rest[6:]

            if ":" in rest:
                # Good :param name: format, use this.
                name_idx = rest.index(":")
            elif " " in rest:
                # Possibly :param name format, use that?
                name_idx = rest.index(" ")
            else:
                # What the hell?
                return description, None

            name = rest[:name_idx].strip()
            rest = rest[name_idx+1:]

            if ":param" in rest:
                # There's more params.
                start = rest.index(":param")
            else:
                match = list(re.finditer(r'(:[\w]+)', rest))
                if match:
                    start = match[0].start()
                else:
                    start = len(rest)

            help = rest[:start].strip()
            # Reduce help to one-liner.
            help = " ".join(help.split("\n"))
            rest = rest[start:]

            params[name] = help

        if rest:
            description += "\n" + rest

        return description, params


def parse_args_for_callable(fn):
    assert inspect.isfunction(fn) or inspect.ismethod(fn)

    sig = inspect.signature(fn)
    docstr = inspect.getdoc(fn).strip()
    docstr, param_helps = _parse_docstr(docstr)
    assert docstr and param_helps, "Please write documentation :)"

    parser = ArgParser()
    for arg_name, arg in sig.parameters.items():
        if arg_name == "self" or arg_name == "logger":
            continue

        # Arguments are required or not required,
        # and have either an annotation or default value.
        # By default, args are parsed as strings if not otherwise specified.
        # REVIEW josephz: Is there a better way to determine if it is a positional/required argument?
        required = arg.default is inspect.Parameter.empty
        default = arg.default if arg.default is not inspect.Parameter.empty else None
        type_ = arg.annotation if arg.annotation is not inspect.Parameter.empty else type(
            default) if default is not None else str

        helpstr = param_helps.get(arg_name, None)

        if type_ is bool:
            add_boolean_argument(parser, arg_name, help=helpstr, default=False if default is None else default)
        elif type_ in (tuple, list, GeneratorType):
            parser.add_argument("--" + arg_name, default=default, type=type_, nargs="+", help=helpstr)
        elif hasattr(type_, "__origin__") and (type_.__origin__ == List or type_.__origin__ == Tuple or type_.__origin__ == list or type_.__origin__ == tuple):
            type_ = type_.__args__[0]
            parser.add_argument("--" + arg_name, default=default, type=type_, nargs="+", help=helpstr)
        elif (hasattr(type_, "__origin__") and type_.__origin__ != List and type_.__origin__ != Tuple and
              ((hasattr(type_.__args__[0], "__origin__") and (type_.__args__[0].__origin__ in (list, tuple))) or
               (hasattr(type_.__args__[1], "__origin__") and (type_.__args__[1].__origin__ in (list, tuple))))):
            nargs = "*" if isinstance(type_, type(Union)) else "+"
            type_ = type_.__args__[1] if type_.__args__[0] is type(None) else type_.__args__[0]
            type_ = type_.__args__[0]
            parser.add_argument("--" + arg_name, default=default, type=type_, nargs=nargs, help=helpstr)
        else:
            parser.add_argument("--" + arg_name, default=default, type=type_, required=required, help=helpstr)

    ArgParser.set_help(docstr)
    argv = parser.parse_args()

    return argv


def test_cmdline(required,
                 required_anno: int,
                 required_anno_tuple: tuple,
                 not_required="Test",
                 not_required_bool=False,
                 not_required_int=123,
                 not_required_float=123.0,
                 not_required_tuple=(1, 2, 3),
                 not_required_str="123",
                 not_required_None=None,
                 ):
    """

    :param required: A required, unannotated parameter.
    :param required_anno: A required, annotated int parameter.
    :param required_anno_tuple: A required, annotated tuple parameter.
    :param not_required: A required, default-string parameter.
    :param not_required_bool: A not-required, default-string parameter.
    :param not_required_int: A not-required, default-int parameter.
    :param not_required_float: A not-required, default-float parameter.
    :param not_required_tuple: A not-required, default-tuple parameter.
    :param not_required_str: A not-required, default-string parameter.
    :param not_required_None: A not-required, default-string parameter.
    """
    print("required:", required)
    print("required_anno: ", required_anno)
    print("required_anno_tuple:", required_anno_tuple)
    print("not_required: ", not_required)
    print("not_required_bool:", not_required_bool)
    print("not_required_int:", not_required_int)
    print("not_required_float:", not_required_float)
    print("not_required_tuple:", not_required_tuple)
    print("not_required_str:", not_required_str)
    print("not_required_None:", not_required_None)


def main():
    args = parse_args_for_callable(test_cmdline)
    varsArgs = vars(args)
    test_cmdline(**varsArgs)


if __name__ == "__main__":
    main()
