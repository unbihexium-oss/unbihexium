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
# Registers tab completion for the `unbihexium` command in Bash. The
# completion offers the subcommands after `unbihexium`, the options of each
# subcommand after its name, and values after the options that take one:
# model names after --model, variants after --variant, output formats after
# --format and file names after --input and --output. Anything else falls
# back to the list of subcommands.
#
# Installation: source the file from ~/.bashrc, for example
#
#   source /path/to/unbihexium/scripts/unbihexium-completion.bash
#
# or copy it to /etc/bash_completion.d/ to enable it for every user.
#
# The word lists below are maintained by hand; update them when commands or
# options of unbihexium.cli change.
# =============================================================================

# COMPREPLY=( $(compgen ...) ) with an unquoted ${cur} is the conventional
# completion idiom: compgen prints one candidate per line and none of the
# candidates contains whitespace, so word splitting is intended here. `opts`
# is declared for compatibility with the classic completion template.
# shellcheck disable=SC2034,SC2086,SC2207

# Fill COMPREPLY with the candidates for the word under the cursor.
_unbihexium_completions() {
    # Current word, previous word and a spare list, local to the function.
    local cur prev opts
    # Start with no candidates.
    COMPREPLY=()
    # Word being completed.
    cur="${COMP_WORDS[COMP_CWORD]}"
    # Word before it, which selects the candidate list.
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Subcommands of unbihexium.
    local commands="detect segment predict analyze export serve info version help"

    # Options of detect.
    local detect_opts="--model --variant --threshold --output --format --tile-size"
    # Options of segment.
    local segment_opts="--model --variant --threshold --output --format --tile-size"
    # Options of predict.
    local predict_opts="--model --variant --input --output --batch-size"
    # Options of analyze.
    local analyze_opts="--type --input --output --stats"
    # Options of export.
    local export_opts="--format --model --output"
    # Options of serve.
    local serve_opts="--host --port --workers --reload"

    # Model zoo variants accepted by --variant.
    local variants="tiny base large mega"

    # Detection models accepted by detect --model.
    local detect_models="ship building aircraft vehicle solar_panel oil_storage"

    # Segmentation models accepted by segment --model.
    local segment_models="water crop forest urban road cloud"

    # Output formats accepted by --format.
    local formats="geotiff cog zarr netcdf shapefile geojson"

    # Choose the candidates from the previous word.
    case "${prev}" in
        # After the command name, offer the subcommands.
        unbihexium)
            COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After detect, offer its options.
        detect)
            COMPREPLY=( $(compgen -W "${detect_opts}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After segment, offer its options.
        segment)
            COMPREPLY=( $(compgen -W "${segment_opts}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After predict, offer its options.
        predict)
            COMPREPLY=( $(compgen -W "${predict_opts}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After analyze, offer its options.
        analyze)
            COMPREPLY=( $(compgen -W "${analyze_opts}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After export, offer its options.
        export)
            COMPREPLY=( $(compgen -W "${export_opts}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After serve, offer its options.
        serve)
            COMPREPLY=( $(compgen -W "${serve_opts}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After --model, offer the models of the subcommand.
        --model)
            # Models of the detect subcommand.
            if [[ "${COMP_WORDS[1]}" == "detect" ]]; then
                COMPREPLY=( $(compgen -W "${detect_models}" -- ${cur}) )  # Words of the list that start with the current word.
            # Models of the segment subcommand.
            elif [[ "${COMP_WORDS[1]}" == "segment" ]]; then
                COMPREPLY=( $(compgen -W "${segment_models}" -- ${cur}) )  # Words of the list that start with the current word.
            fi  # Other subcommands have no model list.
            return 0  # Completion done.
            ;;  # End of this case.
        # After --variant, offer the variants.
        --variant)
            COMPREPLY=( $(compgen -W "${variants}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After --format, offer the formats.
        --format)
            COMPREPLY=( $(compgen -W "${formats}" -- ${cur}) )  # Words of the list that start with the current word.
            return 0  # Completion done.
            ;;  # End of this case.
        # After --input or --output, offer file names.
        --input|--output)
            COMPREPLY=( $(compgen -f -- ${cur}) )  # Candidates: file names matching the word.
            return 0  # Completion done.
            ;;  # End of this case.
        # Any other word: fall through to the subcommands below.
        *)
            ;;  # End of this case.
    esac  # End of the case statement.

    # Default: offer the subcommands.
    COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
    return 0  # Completion done.  # Completion done.
}  # End of _unbihexium_completions.

# Use the function above to complete the unbihexium command.
complete -F _unbihexium_completions unbihexium

# =============================================================================
# End of file scripts/unbihexium-completion.bash
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
