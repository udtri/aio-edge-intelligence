{{/*
=============================================================================
Helm Template Helpers — aio-sensor-intelligence
=============================================================================
Reusable named templates for consistent naming and labeling across all
chart resources.
=============================================================================
*/}}

{{/*
Chart name (from Chart.yaml)
*/}}
{{- define "aio-sensor-intelligence.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name. Uses release name + chart name, truncated to 63
characters (Kubernetes label limit).
*/}}
{{- define "aio-sensor-intelligence.fullname" -}}
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

{{/*
Chart label value (name + version)
*/}}
{{- define "aio-sensor-intelligence.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}
