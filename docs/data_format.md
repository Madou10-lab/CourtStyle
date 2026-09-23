# Input data formats

CourtStyle does not redistribute data. This page lists the files each stage
expects, their columns, and how the code uses them, so that equivalent data can
be prepared. Default locations are those in `configs/serve.yaml` and
`configs/rally.yaml`. All paths are relative to the repository root.

## Serve branch

### Serve records: `data/serve/fingerprint_serve.csv`

One row per server and match, 41 columns. Columns are grouped below by their
role in `courtstyle.data.serve_dataset`.

| Role | Columns |
|---|---|
| Speed | `Avg Speed`, `1st Avg`, `1st StDev`, `1st T Avg`, `1st Wide Avg`, `Max 1st`, `Min 1st`, `2nd Avg`, `2nd StDev`, `2nd T Avg`, `2nd Wide Avg`, `Max 2nd`, `Min 2nd` |
| Placement | `%_serves_T`, `%_serves_Wide`, `%_serves_Body`, `serve_width_var`, `serve_depth_var`, `change_T`, `change_Wide`, `change_Body` |
| Player | `player_ht`, `player_age` |
| Rally context | `avg_rally_len_first_serve`, `avg_rally_len_second_serve`, `%_short_rally_wins_after_serve` |
| Outcomes | `player_ace`, `player_df`, `player_svpt`, `player_1stIn`, `player_1stWon`, `player_2ndWon`, `player_SvGms`, `player_bpSaved`, `player_bpFaced` |
| Surface (one-hot) | `surface_Hard`, `surface_Grass`, `surface_Clay` |
| Handedness (one-hot) | `hand_L`, `hand_R`, `hand_U` |

Preprocessing:

* The nine outcome columns are moved to sidecar files. They are used for
  profiling and validation and are never encoder inputs.
* `player_age` and the three rally-context columns are dropped.
* The directional averages and maxima are reparameterised into
  `1st_headroom`, `1st_delta_T`, `1st_delta_Wide`, `2nd_headroom`,
  `2nd_delta_T` and `2nd_delta_Wide`, and the source columns are dropped.
* Records are split 80/20 into training and validation, stratified by surface.
  Imputation and standardisation are fitted on the training split only.

The processed files `train_data.csv`, `val_data.csv` and
`synthetic_processed.csv` in `data/serve/processed/` have 29 columns:

```
Avg Speed, 1st Avg, 1st StDev, Min 1st, 2nd Avg, 2nd StDev, Min 2nd,
%_serves_T, %_serves_Wide, %_serves_Body, serve_width_var, serve_depth_var,
change_T, change_Wide, change_Body, player_ht,
surface_Grass, hand_L, hand_R, surface_Hard, surface_Clay, hand_U,
1st_headroom, 1st_delta_T, 1st_delta_Wide, 2nd_headroom, 2nd_delta_T, 2nd_delta_Wide, surface_U
```

### Synthetic records: `data/serve/synthetic_serve_50k.csv`

Written by `courtstyle.data.synthetic_serve` from the serve records, with the
same columns. They are used only for stage-1 pre-training.

### Held-out tournament: `data/serve/test_set/australian_open_2017.csv`

One row per server and match of the held-out tournament: the identifier columns
`Player`, `Match` and `Result`, followed by the serve-record columns above.
`hand_U` may be absent. Records are embedded with the frozen encoder and the
fitted preprocessor, then assigned to a style by weighted k-nearest-neighbour
voting.

### Held-out annotations: `data/serve/test_set/australian_open_2017_annotations.csv`

One row per held-out record, aligned with the held-out file by `row_index`
(zero-based row position) and `row_hash`.

| Column | Used by | Content |
|---|---|---|
| `instance_id` | both evaluations | Unique identifier, `ao2017_<row_index as six digits>` |
| `row_index` | both evaluations | Row position in the held-out file |
| `row_hash` | `evaluate_reports` | First 12 hex digits of the MD5 of the string `Player\|Match\|Result\|Avg Speed\|player_ace`, built from the held-out row as read by pandas |
| `Player`, `Match`, `Result` | reference | Copied from the held-out file |
| `speed_tier`, `placement_bias`, `variability`, `risk`, `payoff`, `second_serve_resilience` | `evaluate_serve_heldout` | Annotated style attributes, compared per attribute with the assigned style |
| `style_name_short` | reference | Short style name given by the annotator |
| `coach_note` | `evaluate_reports` | Coach-annotated reference note for the record |
| `annotator_confidence` | reference | Annotator confidence |
| `community_id_gt` | `evaluate_serve_heldout` | Annotated style identifier, compared with the assigned style |

Header template:

```
instance_id,row_index,row_hash,Player,Match,Result,speed_tier,placement_bias,variability,risk,payoff,second_serve_resilience,style_name_short,coach_note,annotator_confidence,community_id_gt
```

## Rally branch

### Tracked shots: `data/rally/shots_merged.csv`

One row per tracked shot. A rally is identified by (`match_id`, `point_index`)
and ordered by `shot_index`. `courtstyle.data.rally_dataset` requires:

| Role | Columns |
|---|---|
| Keys | `match_id`, `point_index`, `shot_index` |
| Players (metadata only) | `player`, `player_opponent`, `player_name`, `player_opponent_name` |
| Shot type | `event_shot`, `stroke`, `spin_type`, `is_serve`, `is_return`, `is_rally`, `is_volley`, `is_groundstroke` |
| Ball | `speed`, `start_speed`, `ball_acceleration`, `spin_rpm`, `distance_travelled`, `hit_peak_z`, `bounce_peak_z`, `net_clearance_z_debug`, `duration`, `bounce`, `rebound_net` |
| Ball positions | `start_pos_x`, `start_pos_y`, `start_pos_z`, `end_pos_x`, `end_pos_y`, `bounce_coordinate_x`, `bounce_coordinate_y` |
| Player positions | `player_position_at_hit_x`, `player_position_at_hit_y`, `player_opponent_position_at_hit_x`, `player_opponent_position_at_hit_y`, `player_position_at_bounce_x`, `player_position_at_bounce_y`, `player_opponent_position_at_bounce_x`, `player_opponent_position_at_bounce_y` |

Player identifiers and names are kept as metadata and are never encoder inputs.
Matches are split whole into training and test partitions, and the training
matches are split again into encoder training and validation matches
(`data/rally/split_info.json`). Missing stroke types are imputed and numeric
features standardised with statistics fitted on the training matches only.

The encoder tokens use 31 raw shot variables. Each token concatenates the
variables of the current shot, the following shot and their difference
(31 × 3 = 93 dimensions).

### Point tables: `data/rally/points/<match_id>_<...>.points.csv`

One file per match. The match identifier is the filename prefix before the
first underscore and must match `match_id` in the shot file. Columns read:

| Role | Columns |
|---|---|
| Point key | `index` (joined to `point_index`) |
| Handedness | `server`, `receiver`, `server_name`, `receiver_name`, `player_1_handeness`, `player_2_handeness` |
| Outcomes | `name`, `result`, `reason`, `is_match_point`, `is_set_point`, `is_break_point`, `is_deciding_point`, `tie_break`, `final_set`, `final_set_tie_break`, `conclusion_stroke`, `conclusion_result`, `conclusion_event_ID`, `conclusion_action`, `num_shots`, `duration` |

Outcomes are written to `train_outcomes.csv` and `test_outcomes.csv` and are
used for profiling, confound checks and the held-out outcome association. They
are never encoder inputs. `result` holds the point result (for example `Played`,
`Fault` or `Let`). The held-out outcome association uses the values present in
the test matches.
Missing outcome columns are filled with empty values.
