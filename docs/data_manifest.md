# Data Manifest

Project: Behavior-Aware EV Battery Health Monitoring and Usage Optimization System  
Owner: Naveen Vaidyanathan  
Created: 03-Jun-2026  
Status: Foundation dataset inventory

---

## 1. Purpose

This document records the battery datasets planned for use in the Behavior-Aware BMS project.

The purpose of this file is to clearly document:

- dataset source
- dataset institution
- source link
- raw data location
- interim data location
- processed data location
- expected outputs
- project usage
- ownership
- current status

This project does not replace an existing Battery Management System. Instead, it builds a software intelligence layer above BMS telemetry to identify usage patterns, estimate degradation risk, and generate battery-life improvement recommendations.

---

## 2. Folder Structure

All datasets will follow this structure:

```text
data/
├── raw/
│   ├── nasa/
│   ├── calce/
│   └── stanford/
│
├── interim/
│   ├── nasa/
│   ├── calce/
│   └── stanford/
│
└── processed/
    ├── nasa/
    ├── calce/
    └── stanford/
