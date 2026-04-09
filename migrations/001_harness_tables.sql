-- Harness trace metadata
create table if not exists harness_traces (
  id                   uuid primary key default gen_random_uuid(),
  job_id               text not null unique,
  step_name            text not null,
  status               text not null,
  attempts             int not null,
  final_score          float,
  cheap_check_failures int default 0,
  total_latency_ms     float,
  inputs_summary       text,
  genome_version       int,
  created_at           timestamptz default now()
);

create index if not exists idx_harness_traces_step
  on harness_traces(step_name, created_at desc);

-- Harness genome versions
create table if not exists harness_genomes (
  id                   uuid primary key default gen_random_uuid(),
  step_name            text not null,
  version              int not null,
  is_active            boolean default false,
  parent_version       int,
  manager_prompt       text,
  rubric               jsonb,
  score_threshold      float,
  max_retries          int,
  agent_system_prompt  text,
  output_format_prompt text,
  optimization_notes   text,
  test_score           float,
  test_details         jsonb,
  created_at           timestamptz default now(),
  unique(step_name, version)
);

create index if not exists idx_harness_genomes_active
  on harness_genomes(step_name) where is_active = true;

-- Storage bucket for full trace JSON payloads
insert into storage.buckets (id, name, public)
values ('harness-traces', 'harness-traces', false)
on conflict (id) do nothing;
