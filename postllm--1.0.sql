create schema postllm;

create function postllm.load_model(text)
returns void
as 'postllm','load_model'
language C strict;

create function postllm.free_model(text)
returns void
as 'postllm','free_model'
language C strict;

create function postllm.prompt_model(text,int,int,text)
returns text
as 'postllm','prompt_model'
language C strict;

create function postllm.text_to_token_length(text,text)
returns int
as 'postllm','text_to_token_length'
language C strict;

create function postllm.text_to_tokens(text,text)
returns int[]
as 'postllm','text_to_tokens'
language C strict;

create function postllm.model_n_ctx(text)
returns int
as 'postllm','model_n_ctx'
language C strict;

create function postllm.parse_json_markdown(text)
returns json as $$
declare
    json_text text;
begin
    if $1 ~ '```(json)?\s*(.*)\s*```' then
        json_text := (regexp_match($1, '```(json)?\s*(.*)\s*```'))[2];
        -- raise exception 'Extracted markdown "%" from "%"', json_text, $1;
    else
        json_text := $1;
        -- raise exception 'Extracted json "%"', $1;
    end if;
    return json_text::json;
end;
$$ language plpgsql;