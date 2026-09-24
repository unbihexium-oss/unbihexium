{{- /*
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : deploy/helm/unbihexium/templates/_helpers.tpl
Title       : Named templates shared by the chart templates
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Helm named templates (Go text/template), read by Helm 3
=============================================================================

Abstract
--------
Names, labels, the image reference and the service account name used by
every template. The names follow the conventions of `helm create`.
=============================================================================
*/ -}}
{{/* Chart name, overridable with nameOverride. */}}
{{- define "unbihexium.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Full name of the release resources, at most 63 characters. */}}
{{- define "unbihexium.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/* Chart name and version for the helm.sh/chart label. */}}
{{- define "unbihexium.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Labels that select the pods of the release. */}}
{{- define "unbihexium.selectorLabels" -}}
app.kubernetes.io/name: {{ include "unbihexium.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: api
{{- end }}

{{/* Labels of every resource. */}}
{{- define "unbihexium.labels" -}}
helm.sh/chart: {{ include "unbihexium.chart" . }}
{{ include "unbihexium.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/* Name of the service account of the pods. */}}
{{- define "unbihexium.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "unbihexium.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/* Image reference: repository with digest when given, otherwise with tag. */}}
{{- define "unbihexium.image" -}}
{{- if .Values.image.digest }}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest }}
{{- else }}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) }}
{{- end }}
{{- end }}

{{/* Name of the Secret that holds the API key, empty when none is used. */}}
{{- define "unbihexium.apiKeySecret" -}}
{{- if .Values.server.apiKey.existingSecret }}
{{- .Values.server.apiKey.existingSecret }}
{{- else if .Values.server.apiKey.value }}
{{- printf "%s-api-key" (include "unbihexium.fullname" .) }}
{{- end }}
{{- end }}

{{- /*
=============================================================================
End of file deploy/helm/unbihexium/templates/_helpers.tpl
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
*/ -}}
