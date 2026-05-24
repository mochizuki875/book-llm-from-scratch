import torch
from torch import Tensor, nn

from llm_from_scratch.transformer.attention import MultiHeadAttention
from llm_from_scratch.transformer.utils import LayerNorm, PositionalEncoding


# エンコーダブロック
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model) # マルチヘッドアテンション
        self.layer_norm1 = LayerNorm(d_model) # レイヤー正規化(標準化)
        self.feed_forward = nn.Sequential( # FFN層
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model) # レイヤー正規化(標準化)

    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None) -> Tensor:

        # MultiHeadAttentionにはquery, key, valueを入力する
        # エンコーダブロックでは全てx(位置情報が加算された埋め込みベクトル)を入力
        x_attention = self.attention(x, x, x, mask=src_padding_mask)
        x = self.layer_norm1(x + x_attention)
        x_ff = self.feed_forward(x)
        x = self.layer_norm2(x + x_ff)
        return x

# エンコーダ
class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model) # 埋め込み層を定義(token IDを埋め込みベクトルに変換) ここでtoken IDと埋め込みベクトルのテーブルが作られる
        self.pe = PositionalEncoding(d_model, max_sequence_len) # 位置エンコーディングベクトルの行列を作成
        self.blocks = nn.ModuleList( # エンコーダブロックをn_blocks個積み重ねる
            [EncoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)]
        )

    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None) -> Tensor:
        x = self.embedding(x) # token IDを埋め込みベクトルに変換
        x = self.pe(x) # 位置エンコーディングベクトルを埋め込みベクトルに加算
        for block in self.blocks: # エンコーダブロックを順番に通す
            x = block(x, src_padding_mask=src_padding_mask)
        return x

# デコーダブロック
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model) # マルチヘッドアテンション
        self.layer_norm1 = LayerNorm(d_model) # レイヤー正規化(標準化)
        self.attention_source_target = MultiHeadAttention(n_heads, d_k, d_v, d_model) # ★ソースターゲットアテンション
        self.layer_norm2 = LayerNorm(d_model) # レイヤー正規化(標準化)
        self.feed_forward = nn.Sequential( # FFN層
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm3 = LayerNorm(d_model) # レイヤー正規化(標準化)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,  # エンコーダの出力 
        tgt_mask: Tensor | None = None,
        src_tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x_attention = self.attention(x, x, x, mask=tgt_mask)
        x = self.layer_norm1(x + x_attention)
        x_attention_source_target = self.attention_source_target(
            x, encoder_output, encoder_output, mask=src_tgt_padding_mask
        ) # ★ソースターゲットアテンションではマルチヘッドアテンションのK,Vにエンコーダの出力を入力する
        x = self.layer_norm2(x + x_attention_source_target)
        x_ff = self.feed_forward(x)
        x = self.layer_norm3(x + x_ff)

        return x # 出力は埋め込みベクトル

# デコーダ
class Decoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model) # 埋め込み層を定義(token IDを埋め込みベクトルに変換) ここでtoken IDと埋め込みベクトルのテーブルが作られる
        self.pe = PositionalEncoding(d_model, max_sequence_len) # 位置エンコーディングベクトルの行列を作成
        self.blocks = nn.ModuleList( # デコーダブロックをn_blocks個積み重ねる
            [DecoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor | None = None,
        src_tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                tgt_mask=tgt_mask,
                src_tgt_padding_mask=src_tgt_padding_mask,
            )
        return x # 出力は埋め込みベクトル

# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.encoder = Encoder( # エンコーダを定義
            src_vocab_size, max_sequence_len, d_model, n_blocks, n_heads, d_k, d_v, d_ff
        )
        self.decoder = Decoder( # デコーダを定義
            tgt_vocab_size, max_sequence_len, d_model, n_blocks, n_heads, d_k, d_v, d_ff
        )
        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False) # 線形層を定義(デコーダの出力を語彙数次元に変換)

    # Transformerの順伝播処理
    # エンコーダとデコーダの入力を受け取って、Transformerの出力(語彙数次元の確率分布)を返す
    def forward(
        self,
        src: Tensor, # エンコーダへの入力(入力文章のtoken IDのテンソル)
        tgt: Tensor, # デコーダへの入力(1つ前の出力文章のtoken IDのテンソル)
        src_mask: Tensor | None = None, # エンコーダーのマルチヘッドアッテンションに適用
        tgt_mask: Tensor | None = None, # デコーダーの1つ目のマルチヘッドアッテンションに適用
        src_tgt_mask: Tensor | None = None, # デコーダーの2つ目のマルチヘッドアッテンションに適用
    ) -> Tensor:
        # エンコーダーの計算
        encoder_output = self.encoder(src, src_padding_mask=src_mask)
        # デコーダーの計算
        decoder_output = self.decoder(
            tgt, encoder_output, tgt_mask=tgt_mask, src_tgt_padding_mask=src_tgt_mask
        )
        # 線形層
        #   デコーダの出力(埋め込みベクトル次元)を語彙数次元(確率分布)に変換
        #   ボキャブラリー数分の確率分布がTransformerの出力になる
        output = self.linear(decoder_output)
        return output

    # Transformerを繰り返し呼び出して文章を1トークンずつ生成する処理
    # 推論モードを指定することで学習時の重みの更新を行わないようにする
    @torch.inference_mode
    def inference(self, src: Tensor, bos_token: int, eos_token: int) -> Tensor:
        tgt_tokens = torch.tensor([[bos_token]]).to(src.device) # 出力トークンの初期化(最初はBOSトークンのみ)

        encoder_output = self.encoder(src) # 入力(トークンIDのテンソル)をエンコーダに通して埋め込みベクトルを取得
        for _ in range(20): # 生成する文章の最大長を20トークンに設定(20トークン以上の文章を生成しないようにする)
            # デコーダに出力トークンのテンソルとエンコーダの出力を入力して、デコーダの出力(埋め込みベクトル)を取得
            decoder_output = self.decoder(tgt_tokens, encoder_output)
            # 埋め込みベクトルを線形層に通して語彙数次元の確率分布(logits)を取得する線形層
            pred = self.linear(decoder_output) 
            # 確率分布の中で最も確率の高いトークンIDを取得(貪欲法)して、次の入力トークンとして使用するためにテンソルに変換
            pred = torch.tensor([[pred[0, -1].argmax().item()]]).to(src.device) 
            # 出力トークンのテンソルに次の入力トークンを追加して更新
            tgt_tokens = torch.cat((tgt_tokens, pred), axis=-1)
            # 生成されたトークンがEOSトークンの場合は文章の生成を終了する
            if pred[0, 0].item() == eos_token: 
                break

        return tgt_tokens
