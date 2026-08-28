-- Chạy một lần trong Supabase SQL editor.
create extension if not exists vector;

create table if not exists friday_memory (
  id           bigserial   primary key,
  fact         text        not null,
  provenance   text        not null check (provenance in ('user', 'tool')),
  embedding    vector(768) not null,
  created_at   timestamptz not null default now(),
  last_used_at timestamptz not null default now()
);

-- Chưa có chỉ mục vector: tìm kiếm tương đồng chạy trong process trên cache RAM
-- ở quy mô hiện tại. Thêm ivfflat khi bảng lên hàng nghìn dòng.
