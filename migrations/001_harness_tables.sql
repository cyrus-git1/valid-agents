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

-- Storage bucket for trace JSONL logs (one file per step per day)
-- Path: harness-traces/{step_name}/{YYYY-MM-DD}.jsonl
insert into storage.buckets (id, name, public)
values ('harness-traces', 'harness-traces', false)
on conflict (id) do nothing;
