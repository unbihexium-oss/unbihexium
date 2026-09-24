#!/bin/bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# File        : scripts/unbihexium-completion.bash
# Title       : Bash completion for the unbihexium command line interface
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Format      : Bash script, sourced by an interactive Bash shell
# =============================================================================
#
# Abstract
# --------
# Registers tab completion for the `unbihexium` command in Bash. The script
# is the Bash completion of Click (the CLI framework of unbihexium): on each
# tab press it runs `unbihexium` with _UNBIHEXIUM_COMPLETE=bash_complete,
# which prints the matching commands, options, choices and file names of the
# installed version. It therefore never lists commands that do not exist.
# The function is the output of
#
#   _UNBIHEXIUM_COMPLETE=bash_source unbihexium
#
# with comments added. Installation: source the file from ~/.bashrc, for
# example `source /path/to/unbihexium/scripts/unbihexium-completion.bash`,
# or copy it to /etc/bash_completion.d/. Bash 4.4 or newer is required.
# =============================================================================

# Complete the current word of an unbihexium command line.
_unbihexium_completion() {
    # Split the answer of the CLI at newlines only.
    local IFS=$'\n'
    # Candidates printed by the CLI.
    local response

    # Ask the installed CLI for the candidates of the current word.
    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD="$COMP_CWORD" _UNBIHEXIUM_COMPLETE=bash_complete "$1")

    # Each line is "type,value".
    for completion in $response; do
        # Split the type from the value.
        IFS=',' read -r type value <<< "$completion"

        # Directory names are completed by Bash.
        if [[ $type == 'dir' ]]; then
            # No own candidates.
            COMPREPLY=()
            # Let Bash complete directory names.
            compopt -o dirnames
        # File names are completed by Bash.
        elif [[ $type == 'file' ]]; then
            # No own candidates.
            COMPREPLY=()
            # Let Bash complete file names.
            compopt -o default
        # Commands, options and choices.
        elif [[ $type == 'plain' ]]; then
            # Add the candidate.
            COMPREPLY+=("$value")
        fi  # End of the type cases.
    done  # End of the candidates.

    # Completion succeeded.
    return 0
}

# Use the function above for the unbihexium command, in the CLI's order.
complete -o nosort -F _unbihexium_completion unbihexium

# =============================================================================
# End of file scripts/unbihexium-completion.bash
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
