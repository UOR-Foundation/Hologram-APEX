/**
 * CLIP BPE Tokenizer - JavaScript Implementation
 *
 * A pure JavaScript implementation of Byte Pair Encoding (BPE) tokenization
 * for CLIP text encoder. Works in browser/WASM environments.
 */

export interface TokenizerConfig {
    bosToken: string;
    eosToken: string;
    unkToken: string;
    padToken: string;
    maxLength: number;
}

export class CLIPTokenizer {
    private vocab: Map<string, number>;
    private bpeMerges: Map<string, number>;
    private config: TokenizerConfig;
    private byteEncoder: Map<number, string>;
    private byteDecoder: Map<string, number>;

    constructor() {
        this.vocab = new Map();
        this.bpeMerges = new Map();
        this.config = {
            bosToken: '<|startoftext|>',
            eosToken: '<|endoftext|>',
            unkToken: '<|endoftext|>',
            padToken: '<|endoftext|>',
            maxLength: 77
        };
        this.byteEncoder = new Map();
        this.byteDecoder = new Map();
        this.initByteEncoder();
    }

    /**
     * Initialize byte-level encoding (GPT-2 style)
     * Maps bytes to unicode characters for BPE processing
     */
    private initByteEncoder() {
        const bytes: number[] = [];

        // Printable ASCII
        for (let i = '!'.charCodeAt(0); i <= '~'.charCodeAt(0); i++) {
            bytes.push(i);
        }
        for (let i = '¡'.charCodeAt(0); i <= '¬'.charCodeAt(0); i++) {
            bytes.push(i);
        }
        for (let i = '®'.charCodeAt(0); i <= 'ÿ'.charCodeAt(0); i++) {
            bytes.push(i);
        }

        let n = 0;
        for (let b = 0; b < 256; b++) {
            if (!bytes.includes(b)) {
                bytes.push(b);
                this.byteEncoder.set(b, String.fromCharCode(256 + n));
                n++;
            } else {
                this.byteEncoder.set(b, String.fromCharCode(b));
            }
        }

        // Create reverse mapping
        this.byteEncoder.forEach((char, byte) => {
            this.byteDecoder.set(char, byte);
        });
    }

    /**
     * Load tokenizer from JSON files
     */
    async loadFromFiles(vocabPath: string, mergesPath: string): Promise<void> {
        // Load vocabulary
        const vocabResponse = await fetch(vocabPath);
        const vocabData = await vocabResponse.json();

        Object.entries(vocabData).forEach(([token, id]) => {
            this.vocab.set(token, id as number);
        });

        // Load BPE merges
        const mergesResponse = await fetch(mergesPath);
        const mergesText = await mergesResponse.text();
        const mergeLines = mergesText.split('\n').slice(1); // Skip header

        mergeLines.forEach((line, index) => {
            const trimmed = line.trim();
            if (trimmed) {
                this.bpeMerges.set(trimmed, index);
            }
        });
    }

    /**
     * Convert text to bytes using byte-level encoding
     */
    private textToBytes(text: string): string {
        const encoder = new TextEncoder();
        const bytes = encoder.encode(text);
        return Array.from(bytes)
            .map(b => this.byteEncoder.get(b) || '')
            .join('');
    }

    /**
     * Get all possible bigram pairs in a word
     */
    private getPairs(word: string[]): Set<string> {
        const pairs = new Set<string>();
        for (let i = 0; i < word.length - 1; i++) {
            pairs.add(`${word[i]} ${word[i + 1]}`);
        }
        return pairs;
    }

    /**
     * Apply BPE merges to a word
     */
    private bpe(token: string): string {
        if (token.length <= 1) return token;

        let word = token.split('');
        let pairs = this.getPairs(word);

        if (pairs.size === 0) {
            return token;
        }

        while (true) {
            // Find the pair with the lowest merge rank
            let minPair: string | null = null;
            let minRank = Infinity;

            pairs.forEach(pair => {
                const rank = this.bpeMerges.get(pair);
                if (rank !== undefined && rank < minRank) {
                    minRank = rank;
                    minPair = pair;
                }
            });

            if (minPair === null) break;

            const [first, second] = (minPair as string).split(' ');
            const newWord: string[] = [];
            let i = 0;

            while (i < word.length) {
                const j = word.indexOf(first, i);
                if (j === -1) {
                    newWord.push(...word.slice(i));
                    break;
                }

                newWord.push(...word.slice(i, j));
                i = j;

                if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
                    newWord.push(first + second);
                    i += 2;
                } else {
                    newWord.push(word[i]);
                    i += 1;
                }
            }

            word = newWord;
            if (word.length === 1) break;
            pairs = this.getPairs(word);
        }

        return word.join(' ');
    }

    /**
     * Tokenize text into token IDs
     */
    encode(text: string): number[] {
        // Lowercase and clean text (CLIP preprocessing)
        text = text.toLowerCase().trim();

        // Convert to byte-level representation
        const byteText = this.textToBytes(text);

        // Split into tokens using whitespace
        const words = byteText.split(/\s+/);

        // Apply BPE to each word
        const bpeTokens: string[] = [];
        words.forEach(word => {
            if (word) {
                const bpeWord = this.bpe(word);
                bpeTokens.push(...bpeWord.split(' '));
            }
        });

        // Convert tokens to IDs
        const bosId = this.vocab.get(this.config.bosToken) || 49406;
        const eosId = this.vocab.get(this.config.eosToken) || 49407;
        const padId = eosId; // Use EOS as padding

        const tokenIds = [bosId];

        bpeTokens.forEach(token => {
            const id = this.vocab.get(token);
            if (id !== undefined && tokenIds.length < this.config.maxLength - 1) {
                tokenIds.push(id);
            }
        });

        // Add EOS token
        if (tokenIds.length < this.config.maxLength) {
            tokenIds.push(eosId);
        }

        // Pad to max length
        while (tokenIds.length < this.config.maxLength) {
            tokenIds.push(padId);
        }

        // Truncate if needed
        return tokenIds.slice(0, this.config.maxLength);
    }

    /**
     * Decode token IDs back to text
     */
    decode(tokenIds: number[]): string {
        // Reverse vocab lookup
        const idToToken = new Map<number, string>();
        this.vocab.forEach((id, token) => {
            idToToken.set(id, token);
        });

        // Convert IDs to tokens
        const tokens: string[] = [];
        const eosId = this.vocab.get(this.config.eosToken) || 49407;

        for (const id of tokenIds) {
            if (id === eosId) break; // Stop at EOS
            const token = idToToken.get(id);
            if (token && token !== this.config.bosToken) {
                tokens.push(token);
            }
        }

        // Join and decode bytes
        const byteStr = tokens.join('');
        const bytes: number[] = [];

        for (const char of byteStr) {
            const byte = this.byteDecoder.get(char);
            if (byte !== undefined) {
                bytes.push(byte);
            }
        }

        // Decode UTF-8
        const decoder = new TextDecoder();
        return decoder.decode(new Uint8Array(bytes));
    }
}
