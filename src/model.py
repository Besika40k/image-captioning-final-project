import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256, train_cnn=False):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.projection = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        if train_cnn:
            for param in resnet.layer4.parameters():
                param.requires_grad_(True)

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        features = self.projection(features)
        features = features.permute(0, 2, 1)
        features = self.bn(features)
        features = features.permute(0, 2, 1)
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.relu(att1 + att2)
        att = self.full_att(att).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, attention_dim, decoder_dim, vocab_size, dropout=0.5):
        super().__init__()
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, decoder_dim, attention_dim)
        self.lstm = nn.LSTMCell(embed_size + embed_size, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward(self, features, captions):
        batch_size = features.size(0)
        seq_len = captions.size(1)
        vocab_size = self.fc.out_features

        embeddings = self.embedding(captions)

        h = torch.zeros(batch_size, self.decoder_dim).to(features.device)
        c = torch.zeros(batch_size, self.decoder_dim).to(features.device)

        outputs = torch.zeros(batch_size, seq_len, vocab_size).to(features.device)

        for t in range(seq_len):
            context, _ = self.attention(features, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            outputs[:, t, :] = self.fc(self.dropout(h))

        return outputs


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, attention_dim=256, decoder_dim=512, dropout=0.5, train_cnn=False):
        super().__init__()
        self.encoder = EncoderCNN(embed_size, train_cnn)
        self.decoder = DecoderRNN(embed_size, attention_dim, decoder_dim, vocab_size, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate(self, image_tensor, vocab, max_len=20):
        self.eval()
        with torch.no_grad():
            features = self.encoder(image_tensor)

            h = torch.zeros(1, self.decoder.decoder_dim).to(image_tensor.device)
            c = torch.zeros(1, self.decoder.decoder_dim).to(image_tensor.device)

            input_token = torch.tensor([vocab.start_idx]).to(image_tensor.device)
            captions = []

            for _ in range(max_len):
                embeddings = self.decoder.embedding(input_token)
                context, _ = self.decoder.attention(features, h)
                lstm_input = torch.cat([embeddings, context], dim=1)
                h, c = self.decoder.lstm(lstm_input, (h, c))
                output = self.decoder.fc(h)
                predicted = output.argmax(dim=1)

                captions.append(predicted.item())

                if predicted.item() == vocab.end_idx:
                    break

                input_token = predicted

            return vocab.denumericalize(captions)
