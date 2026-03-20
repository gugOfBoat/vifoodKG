begin;
drop table if exists public.vqa;
drop table if exists public.kg_triple_catalog;

CREATE TABLE public.image (
  image_id text NOT NULL,
  image_url character varying NOT NULL,
  food_items ARRAY,
  is_checked boolean DEFAULT false,
  image_desc text,
  is_drop boolean NOT NULL DEFAULT false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  CONSTRAINT image_pkey PRIMARY KEY (image_id)
);

create table public.kg_triple_catalog (
  triple_id bigserial primary key,
  subject text not null,
  relation text not null,
  target text not null,
  evidence text,
  source_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint kg_triple_catalog_unique unique (subject, relation, target)
);

create index kg_triple_catalog_subject_idx on public.kg_triple_catalog(subject);
create index kg_triple_catalog_relation_idx on public.kg_triple_catalog(relation);
create index kg_triple_catalog_target_idx on public.kg_triple_catalog(target);

create table public.vqa (
  vqa_id bigserial primary key,
  image_id text not null references public.image(image_id) on delete cascade,
  qtype text not null,
  question text not null,
  choice_a text not null,
  choice_b text not null,
  choice_c text not null,
  choice_d text not null,
  answer varchar(1) not null,
  rationale text,
  triples_used jsonb not null default '[]'::jsonb,
  is_checked boolean not null default false,
  is_drop boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint vqa_answer_check check (answer in ('A', 'B', 'C', 'D')),
  constraint vqa_unique_exact unique (image_id, qtype, question)
);

create index vqa_image_id_idx on public.vqa(image_id);
create index vqa_qtype_idx on public.vqa(qtype);
create index vqa_review_state_idx on public.vqa(is_checked, is_drop);
create index vqa_image_qtype_idx on public.vqa(image_id, qtype);

commit;
