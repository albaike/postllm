create schema postllm_test;

-- set log_min_messages to debug1;

create function postllm_test.test_load_model_missing_file()
returns void as $$
declare
    model_filename text;
begin
    select '/bad/model/filename' into model_filename;

    perform postllm.load_model(model_filename);

    raise exception 'Prompt fail not raised.';
exception
    when others then return;
end;
$$ language plpgsql;

create function postllm_test.test_model_n_ctx(
    model_filename text,
    n_ctx_max int
) returns void as $$
declare
    n_ctx int;
begin
    perform postllm.load_model(model_filename);
    select postllm.model_n_ctx(model_filename) into n_ctx;
    perform postllm.free_model(model_filename);

    if n_ctx != n_ctx_max then
        raise exception 'Model n_ctx: %',
            n_ctx;
    end if;
end;
$$ language plpgsql;

create function postllm_test.test_text_to_token_length(model_filename text)
returns void as $$
declare
    prompt_text_length int;
    prompt_text text;
    prompt_token_length int;
begin
    select 32000 into prompt_text_length;
    select substring(repeat('0', prompt_text_length) from 1 for prompt_text_length) into prompt_text;

    perform postllm.load_model(model_filename);
    select postllm.text_to_token_length(model_filename, prompt_text) into prompt_token_length;
    perform postllm.free_model(model_filename);

    if prompt_token_length != 32002 then
        raise exception 'Text: % chars, % tokens',
            prompt_text_length, prompt_token_length;
    end if;
end;
$$ language plpgsql;

create function postllm_test.test_text_to_tokens(model_filename text)
returns void as $$
declare
    prompt_text_length int;
    prompt_text text;
    prompt_tokens int[];
begin
    select 32000 into prompt_text_length;
    select substring(repeat('0', prompt_text_length) from 1 for prompt_text_length) into prompt_text;

    perform postllm.load_model(model_filename);
    select postllm.text_to_tokens(model_filename, prompt_text) into prompt_tokens;
    perform postllm.free_model(model_filename);

    raise exception 'Text: % chars, % tokens',
        prompt_text_length, array_length(prompt_tokens, 1);
end;
$$ language plpgsql;

create function postllm_test.test_prompt_doesnt_output_empty(
    model_filename text,
    n_ctx_max int,
    n_threads int
) returns void as $$
declare
    prompt_result text;
begin
    perform postllm.load_model(model_filename);
    select postllm.prompt_model(
        model_filename,
        512, n_ctx_max, n_threads,
        'Print the exact text: "SAMPLE TEXT". Do not include the quotation marks or any other text, simply print the enclosed text.\n'
    ) into prompt_result;

    if prompt_result = '' then
        raise exception 'Empty prompt response';
    end if;
    perform postllm.free_model(model_filename);
end;
$$ language plpgsql;

create function postllm_test.test_prompt_multi(
    model_filename text,
    n_ctx_max int,
    n_threads int
) returns void as $$
declare
    prompt_result1 text;
    prompt_result2 text;
    prompt_result3 text;
begin
    perform postllm.load_model(model_filename);

    select postllm.prompt_model(model_filename,64,n_ctx_max,n_threads,'a:\n') into prompt_result1;
    select postllm.prompt_model(model_filename,64,n_ctx_max,n_threads,'Say hi!\n') into prompt_result2;
    select postllm.prompt_model(model_filename,64,n_ctx_max,n_threads,'Sum of all even numbers less than 64\n') into prompt_result3;

    perform postllm.free_model(model_filename);
end;
$$ language plpgsql;

create function postllm_test.test_prompt_kv_size_too_small(model_filename text)
returns void as $$
declare
    prompt_result text;
begin
    perform postllm.load_model(model_filename);

    select postllm.prompt_model(
        model_filename,
        1,1,n_threads,
        'Print the exact text: "SAMPLE TEXT". Do not include the quotation marks or any other text, simply print the enclosed text.\n'
    ) into prompt_result;

    perform postllm.free_model(model_filename);

    raise exception '%', prompt_result;
end;
$$ language plpgsql;

create function postllm_test.test_prompt_json(
    model_filename text,
    n_ctx_max int,
    n_threads int
) returns void as $$
declare
    prompt_result text;
begin
    perform postllm.load_model(model_filename);

    select postllm.prompt_model(
        model_filename,
        128, n_ctx_max, n_threads,
        E'Print a JSON representation of presented data, starting with *"```json"* and ending with *"```"*. '
        || E'Do not generate any other text, you must output ONLY valid JSON. '
        || E'Datatypes should best match the given value (ie convert named numbers to JSON integers).\n\n'
        || E'#Data\n\nPaul Anderson, 23, lives in NYC and makes 5 million dollars annually.\n\n'
        || E'#Result\n\n'
    ) into prompt_result;
    perform postllm.free_model(model_filename);

    if not postllm.parse_json_markdown(prompt_result)::jsonb = '{"name":"Paul Anderson","age":23,"location":"NYC","income":5000000}'::jsonb then
        raise exception 'Invalid json response %', postllm.parse_json_markdown(prompt_result)::text;
    end if;
end;
$$ language plpgsql;

create function postllm_test.run_tests()
returns void as $$
declare
    model_filename text;
    n_ctx_max int;
    n_threads int;
begin
    select '/tmp/gemma-1.1-7b-it.Q2_K.gguf' into model_filename;
    select 8192 into n_ctx_max;
    select 0 into n_threads;

    perform postllm_test.test_load_model_missing_file();
    perform postllm_test.test_model_n_ctx(model_filename,n_ctx_max);
    perform postllm_test.test_text_to_token_length(model_filename);
    perform postllm_test.test_prompt_doesnt_output_empty(model_filename,n_ctx_max,n_threads);
    perform postllm_test.test_prompt_multi(model_filename,n_ctx_max,n_threads);
    perform postllm_test.test_prompt_json(model_filename,n_ctx_max,n_threads);
end;
$$ language plpgsql;

select postllm_test.run_tests();

drop schema postllm_test cascade;