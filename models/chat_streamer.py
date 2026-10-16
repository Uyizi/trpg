"""Incremental UTF-8 decoding without TextStreamer's wait-for-space heuristic."""
from tokenizers.decoders import DecodeStream


class IncrementalTextStreamer:
    def __init__(self, tokenizer, on_text):
        self.tokenizer = tokenizer
        self.on_text = on_text
        self.decoder = DecodeStream(skip_special_tokens=True)
        self.next_is_prompt = True
        self.tokens = []
        self.emitted = ''

    def put(self, value):
        if len(value.shape) > 1:
            if value.shape[0] != 1:
                raise ValueError('Chat streaming supports batch size 1')
            value = value[0]
        if self.next_is_prompt:
            self.next_is_prompt = False
            return
        for token_id in value.tolist():
            self.tokens.append(token_id)
            # The Rust decoder holds incomplete UTF-8 bytes, not entire words.
            chunk = self.decoder.step(self.tokenizer.backend_tokenizer, token_id)
            if chunk:
                self.emitted += chunk
                self.on_text(chunk)

    def end(self):
        # Flush a final incomplete-byte replacement only when generation ends.
        final = self.tokenizer.decode(self.tokens, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
        if final.startswith(self.emitted):
            tail = final[len(self.emitted):]
            if tail:
                self.on_text(tail)
        self.on_text('\n')
