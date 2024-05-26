create schema postllm_test;

-- set log_min_messages to debug1;

create function postllm_test.test_prompt_fails_on_model_file_missing()
returns void as $$
begin
    perform postllm.prompt(
        '/bad/model/filename',
        'EXAMPLE PROMPT',
        512
    );

    raise exception 'Prompt fail not raised.';
exception
    when others then return;
end;
$$ language plpgsql;

create function postllm_test.test_prompt_doesnt_output_empty()
returns void as $$
declare
    prompt_result text;
begin
    select postllm.prompt(
        '/tmp/gemma-1.1-7b-it.Q2_K.gguf',
        'Print the exact text: "SAMPLE TEXT". Do not include the quotation marks or any other text, simply print the enclosed text.\n',
        512
    ) into prompt_result;

    if prompt_result = '' then
        raise exception 'Empty prompt response';
    end if;
end;
$$ language plpgsql;

create function postllm_test.test_prompt_json()
returns void as $$
declare
    prompt_result text;
begin
    select postllm.prompt(
        '/tmp/gemma-1.1-7b-it.Q2_K.gguf',
        'Print a JSON representation of the following data. Do not include surrounding quotation marks, markdownlike ```json...``` or any other text, you must output ONLY valid JSON, again not Markdown or any other markup.\nData:\nPaul Anderson, 23, lives in NYC and makes 5 million dollars annually.\nJSON:\n',
        512
    ) into prompt_result;

    if not postllm.parse_json_markdown(prompt_result)::jsonb = '{"name":"Paul Anderson","age":23,"location":"NYC","income":5000000}'::jsonb then
        raise exception 'Invalid json response %', postllm.parse_json_markdown(prompt_result)::text;
    end if;
end;
$$ language plpgsql;

select postllm_test.test_prompt_fails_on_model_file_missing();
select postllm_test.test_prompt_doesnt_output_empty();
select postllm_test.test_prompt_json();
drop schema postllm_test cascade;