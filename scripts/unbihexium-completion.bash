#!/bin/bash
# Unbihexium CLI Bash Completion Script
# Installation: Add to ~/.bashrc or place in /etc/bash_completion.d/

_unbihexium_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main commands
    local commands="detect segment predict analyze export serve info version help"
    
    # Subcommand options
    local detect_opts="--model --variant --threshold --output --format --tile-size"
    local segment_opts="--model --variant --threshold --output --format --tile-size"
    local predict_opts="--model --variant --input --output --batch-size"
    local analyze_opts="--type --input --output --stats"
    local export_opts="--format --model --output"
    local serve_opts="--host --port --workers --reload"
    
    # Model variants
    local variants="tiny base large mega"
    
    # Detection models
    local detect_models="ship building aircraft vehicle solar_panel oil_storage"
    
    # Segmentation models
    local segment_models="water crop forest urban road cloud"
    
    # Output formats
    local formats="geotiff cog zarr netcdf shapefile geojson"

    case "${prev}" in
        unbihexium)
            COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
            return 0
            ;;
        detect)
            COMPREPLY=( $(compgen -W "${detect_opts}" -- ${cur}) )
            return 0
            ;;
        segment)
            COMPREPLY=( $(compgen -W "${segment_opts}" -- ${cur}) )
            return 0
            ;;
        predict)
            COMPREPLY=( $(compgen -W "${predict_opts}" -- ${cur}) )
            return 0
            ;;
        analyze)
            COMPREPLY=( $(compgen -W "${analyze_opts}" -- ${cur}) )
            return 0
            ;;
        export)
            COMPREPLY=( $(compgen -W "${export_opts}" -- ${cur}) )
            return 0
            ;;
        serve)
            COMPREPLY=( $(compgen -W "${serve_opts}" -- ${cur}) )
            return 0
            ;;
        --model)
            if [[ "${COMP_WORDS[1]}" == "detect" ]]; then
                COMPREPLY=( $(compgen -W "${detect_models}" -- ${cur}) )
            elif [[ "${COMP_WORDS[1]}" == "segment" ]]; then
                COMPREPLY=( $(compgen -W "${segment_models}" -- ${cur}) )
            fi
            return 0
            ;;
        --variant)
            COMPREPLY=( $(compgen -W "${variants}" -- ${cur}) )
            return 0
            ;;
        --format)
            COMPREPLY=( $(compgen -W "${formats}" -- ${cur}) )
            return 0
            ;;
        --input|--output)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        *)
            ;;
    esac

    COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
    return 0
}

complete -F _unbihexium_completions unbihexium
