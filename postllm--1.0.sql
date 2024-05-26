create schema postllm;

create function postllm.load_model(text)
returns void
as 'postllm','load_model'
language C strict;

create function postllm.free_model(text)
returns void
as 'postllm','free_model'
language C strict;

create function postllm.prompt_model(text,text,int)
returns text
as 'postllm','prompt_model'
language C strict;

create function postllm.prompt(text,text,int)
returns text
as 'postllm','prompt'
language C strict;

create function postllm.parse_json_markdown(text)
returns json as $$
declare
    json_text text;
begin
    if $1 ~ '```(json)?\s*(.*)\s*```' then
        json_text := (regexp_match($1, '```(json)?\s*(.*)\s*```'))[2];
        -- raise exception 'Extracted markdown json: "%" from "%"', json_text, $1;
    else
        json_text := $1;
        raise exception 'Extracted json: "%"', $1;
    end if;
    return json_text::json;
end;
$$ language plpgsql;