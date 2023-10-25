# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

# This code was modified based on the https://github.com/facebookresearch/text-adversarial-attack

# @article{guo2021gradientbased,
#   title={Gradient-based Adversarial Attacks against Text Transformers},
#   author={Guo, Chuan and Sablayrolles, Alexandre and Jégou, Hervé and Kiela, Douwe},
#   journal={arXiv preprint arXiv:2104.13733},
#   year={2021}
# }

from data import cfg
from bert_score.utils import get_idf_dict
from datasets import load_dataset
from transformers import AutoModelForCausalLM
import transformers
import csv
import math
import argparse
import math
import jiwer
import numpy as np
import os
import warnings
transformers.logging.set_verbosity(transformers.logging.ERROR)
import time 
import torch
import torch.nn.functional as F

from src.dataset import load_data
from src.utils import bool_flag, get_output_file, print_args, load_gpt2_from_dict

from PIL import Image
import torch
from spipe import StableDiffusionPipeline
from classfier import *


def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)


def bert_score(refs, cands, weights=None):
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens; only works when refs and cands all have equal length (!!!)
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()

def check_classname(adv_text, classname):
    if classname in adv_text:
        return True
    else:
        return False    

def main():

    output_file = get_output_file(cfg['start_index'], cfg['start_index'] + cfg['num_samples'])
    output_file = os.path.join(cfg['adv_samples_folder'], output_file)
    print(f"Outputting files to {output_file}")
    if os.path.exists(output_file):
        print('Skipping batch as it has already been completed.')
        exit()  

    imagenet_classes = cfg['imagenet_classes']
    imagenet_templates = cfg['imagenet_templates']
    inference_steps = cfg['num_inference_steps']
    img_size = cfg['img_size']
    save = cfg['save']
    
    # Load dataset
    text_path = '../imageNet_short_prompt.csv'
    #text_path = '../imageNet_long_prompt.csv'
    
    dataset= load_data(text_path)
    
    # Load tokenizer, model, and reference model
    classfier, _ = clip.load("ViT-B/16")
    zeroshot_weights = zeroshot_classifier(classfier,imagenet_classes, imagenet_templates)
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to('cuda')
    pipe.enable_xformers_memory_efficient_attention()  
    tokenizer = pipe.tokenizer
    tokenizer.model_max_length = 77
    text_encoder = pipe.text_encoder
    # scheduler = pipe.scheduler
    # print(scheduler.config)
    generator = torch.Generator("cuda").manual_seed(1024)

    unet = pipe.unet
    vae = pipe.vae
    for _, param in unet.named_parameters():
            param.requires_grad = False
    for _, param in text_encoder.named_parameters():
        param.requires_grad = False
    for _, param in vae.named_parameters():
        param.requires_grad = False    

    ref_model = AutoModelForCausalLM.from_pretrained('../webtext/save_path/clm/checkpoint-94000', output_hidden_states=True).cuda()
    
    with torch.no_grad():
        embeddings = text_encoder.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
        ref_embeddings = ref_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda()).to(torch.float16)
    
    # encode dataset using tokenizer
    text_key = 'prompt'
    testset_key = 'train'
    preprocess_function = lambda examples: tokenizer(examples[text_key], padding="max_length", truncation=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    #print(tokenizer.decode(encoded_dataset['train']['input_ids'][0]))

    # Compute idf dictionary for BERTScore
    if cfg['constraint'] == "bertscore_idf":
        idf_dict = get_idf_dict(dataset['train'][text_key], tokenizer, nthreads=20)

    adv_log_coeffs, clean_texts, adv_texts = [], [], []
    clean_logits = []
    adv_logits = []
    token_errors = []
    times = []
    
    assert cfg['start_index'] < len(encoded_dataset[testset_key]), 'Starting index %d is larger than dataset length %d' % (cfg['start_index'], len(encoded_dataset[testset_key]))
    end_index = min(cfg['start_index'] + cfg['num_samples'], len(encoded_dataset[testset_key]))
    adv_losses, ref_losses, perp_losses, entropies = torch.zeros(end_index - cfg['start_index'], cfg['num_iters']), torch.zeros(end_index - cfg['start_index'], cfg['num_iters']), torch.zeros(end_index - cfg['start_index'], cfg['num_iters']), torch.zeros(end_index - cfg['start_index'], cfg['num_iters'])
    

    torch.autograd.set_detect_anomaly(True)
    for idx in range(cfg['start_index'], end_index):
        clean_text = encoded_dataset[testset_key]['prompt'][idx]
        input_ids = encoded_dataset[testset_key]['input_ids'][idx]
        input_ids_tensor = torch.LongTensor(input_ids).unsqueeze(0).cuda()
        prompt_embeddings = text_encoder(input_ids = input_ids_tensor)[0]
        prompt_embeddings = prompt_embeddings.to(dtype=text_encoder.dtype)
        label = encoded_dataset[testset_key]['label'][idx]
        with torch.no_grad():
            images = pipe(prompt_embeds=prompt_embeddings, num_inference_steps= inference_steps, generator = generator, height=img_size, width=img_size).images
            images_tensor = img_process(images[0], img_size)
            image_features = classfier.encode_image(images_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            clean_logit = 100. * image_features @ zeroshot_weights

        if clean_logit.argmax() != label:
            print('skip this text')
            continue
        classname = encoded_dataset[testset_key]['classname'][idx]
        if save:
            class_filename = clean_filename(clean_text)
            save_pil_image(images[1], class_filename, 'clean', classname) 

        print('LABEL')
        print(label)
        print('TEXT')
        print(tokenizer.decode(input_ids, skip_special_tokens = True))
        # print('LOGITS')
        # print(clean_logit)
        print('CLASSNAME')
        print(classname)


        forbidden = np.zeros(len(input_ids)).astype('bool')
        
        # set [CLS] and [SEP] tokens to forbidden
        #unchange_ids = tokenizer(encoded_dataset[testset_key]['prompt'][idx],truncation=True)['input_ids']
        unchange_ids = tokenizer(clean_text,truncation=True)['input_ids']
        for i, ids in enumerate(input_ids):
            if (ids in unchange_ids[:-1]):# or (ids == 0):
                forbidden[i] = True

        print('unchange_ids: ',unchange_ids)
        print('input_ids: ',input_ids)
        print('forbidden: ',forbidden)


        forbidden_indices = np.arange(0, len(input_ids))[forbidden]
        forbidden_indices = torch.from_numpy(forbidden_indices).cuda()


        start_time = time.time()
        with torch.no_grad():
            orig_output = ref_model(torch.LongTensor(input_ids).cuda().unsqueeze(0)).hidden_states[cfg['embed_layer']]
            if cfg['constraint'].startswith('bertscore'):
                if cfg['constraint'] == "bertscore_idf":
                    ref_weights = torch.FloatTensor([idf_dict[idx] for idx in input_ids]).cuda()
                    ref_weights /= ref_weights.sum()
                else:
                    ref_weights = None

        log_coeffs = torch.zeros(len(input_ids), embeddings.size(0))#.to(torch.float16)
        indices = torch.arange(log_coeffs.size(0)).long()
        log_coeffs[indices, torch.LongTensor(input_ids)] = cfg['initial_coeff']
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad = True
        optimizer = torch.optim.Adam([log_coeffs], lr=cfg['lr'])

        start = time.time()
        for i in range(cfg['num_iters']):
            optimizer.zero_grad()
            coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0).repeat(cfg['batch_size'], 1, 1), hard=False) # B x T x V
            with torch.autocast(device_type='cuda', dtype=torch.float16):
               
                inputs_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D

                # The text encoder here does not accept input in the form of 'text_embeds'. 
                # Therefore, the source code in the 'transformer' library needs to be modified. The method of modification is demonstrated in 'm_clip.py'."
                inputs_embeds = text_encoder(inputs_embeds = inputs_embeds)[0]
                images = pipe(prompt_embeds=inputs_embeds, num_inference_steps=inference_steps, generator = generator, height=img_size, width=img_size).images
                images_tensor = img_process(images[0], img_size)
                image_features = classfier.encode_image(images_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                pred = 100. * image_features @ zeroshot_weights
                
                if cfg['adv_loss'] == 'ce':
                    adv_loss = -F.cross_entropy(pred, label * torch.ones(cfg['batch_size']).long().cuda())
                elif cfg['adv_loss'] == 'cw':
                    top_preds = pred.sort(descending=True)[1]
                    correct = (top_preds[:, 0] == label).long()
                    indices = top_preds.gather(1, correct.view(-1, 1))
                    adv_loss = (pred[:, label] - pred.gather(1, indices).squeeze() + cfg['kappa']).clamp(min=0).mean()
                
                # Similarity constraint
                ref_embeds = (coeffs @ ref_embeddings[None, :, :])
                pred = ref_model(inputs_embeds=ref_embeds)
                if cfg['lam_sim'] > 0:
                    output = pred.hidden_states[cfg['embed_layer']]
                    if cfg['constraint'].startswith('bertscore'):
                        ref_loss = -cfg['lam_sim'] * bert_score(orig_output, output, weights=ref_weights).mean()
                    else:
                        output = output[:, -1]
                        cosine = (output * orig_output).sum(1) / output.norm(2, 1) / orig_output.norm(2, 1)
                        ref_loss = cfg['lam_sim'] * cosine.mean()
                else:
                    ref_loss = torch.Tensor([0]).cuda()
                
                # (log) perplexity constraint
                if cfg['lam_perp'] > 0:
                    perp_loss = cfg['lam_perp'] * log_perplexity(pred.logits, coeffs)
                else:
                    perp_loss = torch.Tensor([0]).cuda()
                
            # Compute loss and backward
            total_loss = adv_loss + ref_loss + perp_loss 
            with torch.autograd.detect_anomaly():
                total_loss.backward()
            
            entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))
            if i % cfg['print_every'] == 0:
                print('Iteration %d: loss = %.4f, adv_loss = %.4f, ref_loss = %.4f, perp_loss = %.4f, entropy=%.4f, time=%.2f' % (
                    i+1, total_loss.item(), adv_loss.item(), ref_loss.item(), perp_loss.item(), entropy.item(), time.time() - start))

            # Gradient step
            log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
            optimizer.step()

            # Log statistics
            adv_losses[idx - cfg['start_index'], i] = adv_loss.detach().item()
            ref_losses[idx - cfg['start_index'], i] = ref_loss.detach().item()
            perp_losses[idx - cfg['start_index'], i] = perp_loss.detach().item()
            entropies[idx - cfg['start_index'], i] = entropy.detach().item()
        times.append(time.time() - start_time)
        
        print('CLEAN TEXT')        
        clean_text = tokenizer.decode(input_ids, skip_special_tokens = True)
        clean_texts.append(clean_text)
        print(clean_text)

        clean_logits.append(clean_logit.detach())
        print('ADVERSARIAL TEXT')
        with torch.autocast(device_type='cuda', dtype=torch.float16), torch.no_grad():
            for j in range(cfg['gumbel_samples']):
                adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)
                adv_ids = adv_ids.cpu().tolist()
                adv_text = tokenizer.decode(adv_ids, skip_special_tokens = True)
                print(adv_text)
                x = tokenizer(adv_text, truncation=True, return_tensors='pt')
                token_errors.append(wer(adv_ids, x['input_ids'][0]))
                images = pipe(adv_text, num_inference_steps=inference_steps, generator = generator, height=img_size, width=img_size).images
                images_tensor = img_process(images[0], img_size)
                image_features = classfier.encode_image(images_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                adv_logit = 100. * image_features @ zeroshot_weights
        
                if adv_logit.argmax() != label and check_classname(adv_text, classname): #or j == cfg['gumbel_samples'] - 1:
                    successful = 'success'
                    adv_texts.append(adv_text)
                    adv_logits.append(adv_logit)
                    if save:
                        save_pil_image(images[1], class_filename, successful, str(j))
                else:
                    successful = 'failed'
                    if save:
                        save_pil_image(images[1], class_filename, successful, str(j))

                if save:
                    if os.path.exists("adv_text.csv"):
                        with open("adv_text.csv", 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([class_filename, successful, j, adv_text])
                    else:
                        with open("adv_text.csv", 'w', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow(['clean_text', 'successful', 'j', 'adv_text'])
                            csv_writer.writerow([class_filename, successful, j, adv_text])
        
                
        # remove special tokens from adv_log_coeffs
        adv_log_coeffs.append(log_coeffs.cpu()) # size T x V
            
    print("Token Error Rate: %.4f (over %d tokens)" % (sum(token_errors) / len(token_errors), len(token_errors)))
    torch.save({
        'adv_log_coeffs': adv_log_coeffs, 
        'adv_logits': torch.cat(adv_logits, 0), # size N x C
        'adv_losses': adv_losses,
        'adv_texts': adv_texts,
        'clean_logits': torch.cat(clean_logits, 0), 
        'clean_texts': clean_texts, 
        'entropies': entropies,
        'labels': list(encoded_dataset[testset_key]['label'][cfg['start_index']:end_index]), 
        'perp_losses': perp_losses,
        'ref_losses': ref_losses,
        'times': times,
        'token_error': token_errors,
    }, output_file)


if __name__ == "__main__":
    main()