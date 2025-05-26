# utils/InternOmni/custom_chat_logic.py

import torch
import warnings
import importlib
import sys
import logging 
from transformers import GenerationConfig
from typing import Optional, List, Tuple, Union, Dict # For type hints

logger = logging.getLogger(__name__) # Use standard logging

# --- Helper to dynamically load get_conv_template (cached) ---
_cached_get_conv_template = {} 

def _get_dynamic_conv_template_function_for_model(model_instance):
    model_class_module_str = model_instance.__class__.__module__
    if model_class_module_str in _cached_get_conv_template:
        if _cached_get_conv_template[model_class_module_str] is None: # Previously failed
             raise RuntimeError(f"Previously failed to load 'get_conv_template' for {model_class_module_str}. Check model installation and trust_remote_code setup.")
        return _cached_get_conv_template[model_class_module_str]

    if model_class_module_str.startswith("transformers_modules."):
        module_parts = model_class_module_str.split('.')
        if len(module_parts) >= 5 : 
            base_module_path = ".".join(module_parts[:-1]) 
            conversation_module_full_path = base_module_path + ".conversation"
            try:
                conversation_module = importlib.import_module(conversation_module_full_path)
                func = getattr(conversation_module, "get_conv_template")
                _cached_get_conv_template[model_class_module_str] = func
                logger.info(f"Dynamically loaded get_conv_template from {conversation_module_full_path}")
                return func
            except Exception as e:
                _cached_get_conv_template[model_class_module_str] = None # Cache failure
                logger.error(f"Could not dynamically load get_conv_template from {conversation_module_full_path}: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load 'get_conv_template' from {conversation_module_full_path}. "
                                   "Custom chat function cannot proceed.") from e
    
    _cached_get_conv_template[model_class_module_str] = None # Cache failure
    logger.error(f"Cannot determine dynamic module path for get_conv_template from model module: {model_class_module_str}")
    raise RuntimeError("Model module path not recognized for dynamic import of 'get_conv_template'.")


def custom_intern_omni_audio_chat(
    model_instance: torch.nn.Module,
    tokenizer,
    pixel_values: Optional[torch.FloatTensor],
    audio: Optional[Dict[str, torch.Tensor]],
    question: Optional[str],
    generation_config: Union[Dict, GenerationConfig],
    history: Optional[List[Tuple[str, str]]] = None,
    return_history: bool = False,
    num_patches_list: Optional[List[int]] = None,
    IMG_START_TOKEN: str = '<img>', 
    IMG_END_TOKEN: str = '</img>', 
    IMG_CONTEXT_TOKEN: str = '<IMG_CONTEXT>',
    AUDIO_START_TOKEN: str = '<audio>', 
    AUDIO_END_TOKEN: str = '</audio>',
    AUDIO_CONTEXT_TOKEN: str = '<AUDIO_CONTEXT>',
    verbose: bool = False
) -> Union[str, Tuple[str, List[Tuple[str, str]]]]:

    logger.debug("[Custom InternOmni Audio Chat] Initiating custom chat logic.")
    
    get_conv_template_fn = _get_dynamic_conv_template_function_for_model(model_instance)

    current_question_for_template = question if question is not None else ""
    original_question_for_history = str(current_question_for_template) # Ensure it's a string for history

    gen_pixel_values = pixel_values
    gen_audio_values = None
    gen_audio_len_after_cnn = None
    gen_audio_token_num = None

    has_actual_audio = False
    if audio and isinstance(audio, dict) and \
       audio.get('audio_values') is not None and audio['audio_values'].numel() > 0 and \
       audio.get('audio_token_num') is not None and audio['audio_token_num'].numel() > 0 and \
       audio.get('audio_len_after_cnn') is not None and audio['audio_len_after_cnn'].numel() > 0:
        has_actual_audio = True

    has_actual_pixels = False
    if pixel_values is not None and pixel_values.numel() > 0:
        has_actual_pixels = True
    
    if history is None:
        temp_q_text = current_question_for_template
        current_question_for_template = ""
        # Order based on your confirmed "correct" Audio_chat: audio placeholder first, then image.
        if has_actual_audio:
            current_question_for_template += f"{AUDIO_START_TOKEN}\n"
        if has_actual_pixels:
            current_question_for_template += f"{IMG_START_TOKEN}\n"
        current_question_for_template += temp_q_text
        
    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if has_actual_pixels else []
    
    if has_actual_pixels:
        expected_sum = sum(num_patches_list) if num_patches_list else 0
        current_pixel_len = len(pixel_values) if pixel_values is not None else 0
        if not (current_pixel_len == expected_sum): # Simplified check
             warnings.warn(f"CustomChat: Pixel values length {current_pixel_len} "
                           f"mismatch with sum of num_patches_list {expected_sum}. "
                           f"Num_patches_list: {num_patches_list}. Recalculating to use total pixel_values length.", UserWarning)
             if pixel_values is not None: num_patches_list = [current_pixel_len]
    elif not has_actual_pixels and num_patches_list and sum(num_patches_list) > 0:
        num_patches_list = []

    # Set context token IDs on the model_instance because model_instance.generate uses them
    model_instance.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model_instance.audio_context_token_id = tokenizer.convert_tokens_to_ids(AUDIO_CONTEXT_TOKEN)

    template = get_conv_template_fn(model_instance.template) # Use model_instance.template
    template.system_message = model_instance.system_message # Use model_instance.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    current_history_for_template = [] if history is None else list(history)
    for (old_q, old_a) in current_history_for_template:
        template.append_message(template.roles[0], old_q)
        template.append_message(template.roles[1], old_a)
    template.append_message(template.roles[0], current_question_for_template)
    template.append_message(template.roles[1], None)
    query_with_placeholders = template.get_prompt()
    final_query = str(query_with_placeholders) # Ensure it's a string

    _num_image_token_per_tile = getattr(model_instance, 'num_image_token', 1)
    if not isinstance(_num_image_token_per_tile, int) or _num_image_token_per_tile <= 0:
        warnings.warn(f"Model 'num_image_token' is invalid ({_num_image_token_per_tile}). Defaulting to 1.", UserWarning)
        _num_image_token_per_tile = 1

    if has_actual_pixels:
        total_img_context_tokens_needed = sum(_num_image_token_per_tile * num_p for num_p in num_patches_list)
        if total_img_context_tokens_needed > 0:
            all_image_tokens_str = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_img_context_tokens_needed + IMG_END_TOKEN
            if f"{IMG_START_TOKEN}\n" in final_query:
                final_query = final_query.replace(f"{IMG_START_TOKEN}\n", all_image_tokens_str + "\n", 1)
            elif IMG_START_TOKEN in final_query:
                final_query = final_query.replace(IMG_START_TOKEN, all_image_tokens_str, 1)
            elif '<image>' in final_query: # Fallback for a generic placeholder
                 final_query = final_query.replace('<image>', all_image_tokens_str, 1)
            else: warnings.warn("Image data present, but placeholder ('<img>' or '<image>') not found.", UserWarning)
        else:
            final_query = final_query.replace(f'{IMG_START_TOKEN}\n', '').replace(IMG_START_TOKEN, '').replace('<image>\n', '').replace('<image>', '')
    else:
        final_query = final_query.replace(f'{IMG_START_TOKEN}\n', '').replace(IMG_START_TOKEN, '').replace('<image>\n', '').replace('<image>', '')

    if has_actual_audio:
        # **CRITICAL CORRECTION for prompt string to match all audio features**
        total_audio_context_tokens_needed = torch.sum(audio['audio_token_num']).item()
        if total_audio_context_tokens_needed > 0:
            audio_tokens_str = AUDIO_START_TOKEN + AUDIO_CONTEXT_TOKEN * total_audio_context_tokens_needed + AUDIO_END_TOKEN
            if f"{AUDIO_START_TOKEN}\n" in final_query:
                final_query = final_query.replace(f"{AUDIO_START_TOKEN}\n", audio_tokens_str + "\n", 1)
            elif AUDIO_START_TOKEN in final_query:
                final_query = final_query.replace(AUDIO_START_TOKEN, audio_tokens_str, 1)
            elif '<audio>' in final_query:
                final_query = final_query.replace('<audio>', audio_tokens_str, 1)
            else: warnings.warn("Audio data present, but placeholder ('<audio>') not found.", UserWarning)
        else:
            final_query = final_query.replace(f'{AUDIO_START_TOKEN}\n', '').replace(AUDIO_START_TOKEN, '').replace('<audio>\n', '').replace('<audio>', '')
        
        gen_audio_values = audio['audio_values'].to(model_instance.device, dtype=model_instance.dtype)
        gen_audio_len_after_cnn = audio['audio_len_after_cnn'].to(model_instance.device)
        gen_audio_token_num = audio['audio_token_num'].to(model_instance.device)
    else:
        final_query = final_query.replace(f'{AUDIO_START_TOKEN}\n', '').replace(AUDIO_START_TOKEN, '').replace('<audio>\n', '').replace('<audio>', '')

    model_inputs = tokenizer(final_query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model_instance.device)
    attention_mask = model_inputs['attention_mask'].to(model_instance.device)
    
    current_generation_config_dict = generation_config if isinstance(generation_config, dict) else generation_config.to_dict()
    final_gen_config_obj = GenerationConfig(**current_generation_config_dict)
    final_gen_config_obj.eos_token_id = eos_token_id

    if verbose:
        logger.info(f"CustomChat - Final query for tokenizer (first 300 chars): {final_query[:300]}")
        logger.info(f"CustomChat - Input IDs shape: {input_ids.shape}")

    # This calls the InternVLChatAudioModel's .generate() method
    # Ensure that .generate() method is also correct (passes input_ids to language_model.generate)
    generation_output_ids = model_instance.generate(
        pixel_values=gen_pixel_values,
        audio_values=gen_audio_values,
        audio_len_after_cnn=gen_audio_len_after_cnn,
        audio_token_num=gen_audio_token_num,
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=final_gen_config_obj
    )
    
    response_text = tokenizer.batch_decode(generation_output_ids, skip_special_tokens=True)[0]
    
    response_text = response_text.split(template.sep.strip())[0].strip()
    if hasattr(template, 'sep2') and template.sep2 and template.sep2 in response_text:
        response_text = response_text.split(template.sep2.strip())[0].strip()

    updated_history = current_history_for_template
    updated_history[-1] = (original_question_for_history, response_text) 

    if return_history:
        return response_text, updated_history
    else:
        if verbose:
            query_to_print = final_query.replace(IMG_CONTEXT_TOKEN, '').replace(AUDIO_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{AUDIO_START_TOKEN}{AUDIO_END_TOKEN}', '<audio>')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            logger.info(f"CustomChat - Formatted PROMPT:\n{query_to_print}\nRESPONSE:\n{response_text}")
        return response_text
