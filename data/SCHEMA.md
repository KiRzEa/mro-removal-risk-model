# Mock dataset schema

One row is one snapshot of an installed component every 30 flight cycles. Features describe the past only; the label looks at the next 30 cycles.

| Column | Type | Unit | Can be missing | Description |
|---|---|---|---|---|
| component_id | string |  | no | Serial of the installed component; a replacement gets a new id |
| slot_id | string |  | no | Position on the aircraft (aircraft and component type) |
| aircraft_id | string |  | no | Aircraft tail identifier |
| snapshot_date | date |  | no | Date the snapshot is taken; features use data before this date |
| component_type | category |  | no | hydraulic_pump, fuel_pump, starter_generator, bleed_air_valve, brake_assembly |
| part_number | category |  | no | Part number variant of the component |
| aircraft_type | category |  | no | A320, B737 or ATR72 |
| operator | category |  | no | Operating airline: OP_A, OP_B or OP_C |
| aircraft_age_years | float | years | no | Age of the aircraft at the snapshot |
| flight_cycles_since_new | int | cycles | no | Cycles flown by this component since it was new |
| flight_hours_since_new | float | hours | no | Flight hours of this component since it was new |
| cycles_since_overhaul | int | cycles | no | Cycles since the last scheduled overhaul |
| flight_hours_last_30_cycles | float | hours | no | Flight hours in the last 30 cycles (utilization) |
| vibration_mm_s_mean | float | mm/s | yes | Mean vibration over the last 30 cycles |
| vibration_mm_s_std | float | mm/s | yes | Standard deviation of vibration over the last 30 cycles |
| oil_temp_c_mean | float | deg C | yes | Mean oil temperature over the last 30 cycles |
| oil_temp_c_std | float | deg C | yes | Standard deviation of oil temperature |
| pressure_kpa_mean | float | kPa | yes | Mean pressure over the last 30 cycles |
| pressure_kpa_std | float | kPa | yes | Standard deviation of pressure |
| current_a_mean | float | A | yes | Mean current over the last 30 cycles |
| current_a_std | float | A | yes | Standard deviation of current |
| fault_count_last_30_cycles | int | count | no | Fault messages in the last 30 cycles |
| distinct_fault_codes_last_30_cycles | int | count | no | Different fault codes in the last 30 cycles |
| last_fault_code | category |  | no | Most recent fault code, or NONE |
| cycles_since_last_fault | int | cycles | no | Cycles since the last fault message, capped at 900 |
| maintenance_actions_last_90_days | int | count | no | Maintenance actions logged in the last 90 days |
| days_since_last_maintenance | int | days | no | Days since the last maintenance action, capped at 365 |
| prior_unscheduled_removals_in_slot | int | count | no | Earlier unscheduled removals of components in this slot |
| removal_within_30_cycles | int | 0/1 | no | 1 if the component is removed without a plan within the next 30 cycles |
