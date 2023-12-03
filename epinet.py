import haiku as hk
import jax
from jax import numpy as jnp
from typing import Sequence, Optional, Callable
import functools
import numpy as np
import chex
import dill

import sys
sys.path.append("/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/")

from enn import losses
from enn import networks
from enn import base
from enn import utils
from enn import datasets

class FrozenLinearLayer(hk.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        weight, 
        bias=hk.initializers.Constant(0.0)):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        w = hk.get_parameter(
            "w", shape=(self.input_size, self.output_size), init=self.weight)
        b = hk.get_parameter("b", shape=(1, self.output_size), init=self.bias)
        w = jax.lax.stop_gradient(w)
        b = jax.lax.stop_gradient(b)
        y = jnp.dot(x, w) + b

        return y

class MatrixInitializer(hk.initializers.Initializer):
    def __inMatrixInitializerit__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, shape, dtype):
        return self.weight

### 1. create base network for Llama-2 
###          (simplification: identity matrix, receiving dola features from DoLA-enhanced model)

def projection_layer(x, feature_size, logit_size, vocab_head_weight):
    vocab_head = FrozenLinearLayer(
                        input_size=feature_size, 
                        output_size=logit_size, 
                        weight=MatrixInitializer(vocab_head_weight))
    return vocab_head(x)


### 2. create epinet for the whole enn
class MLPEpinetWithTrainableAndPrior(networks.epinet.EpinetWithState):
  """MLP epinet with matching prior function."""
  def __init__(self,
               projection_layer,
               index_dim: int,
               num_classes: int,
               epinet_hiddens: Sequence[int],
               prior_epinet_hiddens: Optional[Sequence[int]] = None,
               prior_scale: float = 1):
    """Defines an MLP epinet with matching prior function."""
    if prior_epinet_hiddens is None:
      prior_epinet_hiddens = epinet_hiddens

    def epinet_fn(hidden: chex.Array,
                  index: base.Index) -> networks.base.OutputWithPrior:
      # Creating networks
      train_epinet = networks.ProjectedMLP(
          epinet_hiddens, num_classes, index_dim, name='train_epinet')
      prior_epinet = networks.ProjectedMLP(
          prior_epinet_hiddens, num_classes, index_dim, name='prior_epinet')

      epi_inputs = hidden

      # Wiring networks: add linear epinet (+ prior) from final output layer.
    #   print("epi_inputs: ", epi_inputs.shape)
      epi_train_logits = projection_layer(train_epinet(epi_inputs, index))
      epi_prior_logits = projection_layer(prior_epinet(epi_inputs, index))
      return networks.OutputWithPrior(
          train=epi_train_logits,
          prior=prior_scale * epi_prior_logits,
      )

    # Form ENN from haiku transformed.
    transformed = hk.without_apply_rng(hk.transform_with_state(epinet_fn))
    indexer = networks.GaussianIndexer(index_dim)

    super().__init__(transformed.apply, transformed.init, indexer)


class XentLoss(losses.SingleLossFnArray):
    """Cross-entropy single index loss with network state as auxiliary."""

    def __init__(self, num_classes: int):
        assert num_classes > 1
        super().__init__()
        self.num_classes = num_classes
        labeller = lambda x: jax.nn.one_hot(x, self.num_classes)
        self._loss = self.xent_loss_with_dola_distributions(labeller)

    def __call__(
        self,
        apply: networks.ApplyArray,
        params: hk.Params,
        state: hk.State,
        batch: datasets.ArrayBatch,
        index: base.Index,
    ) -> base.LossOutput:
        return self._loss(apply, params, state, batch, index)

    def xent_loss_with_dola_distributions(self,
        labeller: Callable[[chex.Array], chex.Array]
    ) -> losses.SingleLossFnArray:
        """Factory method to create a loss function with custom labelling."""

        def single_loss(
            apply: networks.ApplyArray,
            params: hk.Params,
            state: hk.State,
            batch: datasets.ArrayBatch,
            index: base.Index,
        ) -> base.LossOutput:
            """Xent loss with custom labelling."""
            chex.assert_shape(batch.y, (None, 1))
            net_out, state = apply(params, state, batch.x, index)
            logits = networks.parse_net_output(net_out)
            labels = labeller(batch.y[:, 0])

            # combine with dola distributions
            # logits.shape = [batch_size, num_classes]
            # combined_logits = jax.nn.softmax(logits) + jax.lax.stop_gradient(batch.extra['dola_distribution'])
            combined_logits = logits + jax.lax.stop_gradient(batch.extra['dola_distribution'])
            # combined_logits = logits
            softmax_xent = -jnp.sum(
                labels * jax.nn.log_softmax(combined_logits), axis=1, keepdims=True)

            if batch.weights is None:
                batch_weights = jnp.ones_like(batch.y)
            else:
                batch_weights = batch.weights
            chex.assert_equal_shape([batch_weights, softmax_xent])

            loss = jnp.mean(batch_weights * softmax_xent)
            return loss, (state, {'loss': loss})
        return single_loss


# create dataset for training
def get_dummy_dataset(input_dim, num_classes, num_batch, batch_size):
    seed = 0
    # (num_batch * batch_size, input_dim)
    x = np.random.RandomState(seed).randn(input_dim, num_batch * batch_size).T
    y = np.random.RandomState(seed).randint(0,num_classes, num_batch * batch_size)
    # (num_batch * batch_size, num_classes)
    dola_distribution = np.random.RandomState(seed).randn(num_classes, num_batch * batch_size).T
    dola_distribution = jax.nn.softmax(dola_distribution)
    # # print(x.shape, y.shape, dola_distribution.shape)

    # # Load the actual DoLa dataset for epinet training
    # feats_actual = torch.load('/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/data/dola_data_test/CSE8803-DLT/C4_data_100samples/layer_features.pt')
    # dola_actual = torch.load('/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/data/dola_data_test/CSE8803-DLT/C4_data_100samples/dola_output_logits.pt')
    # labels_actual = torch.load('/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/data/dola_data_test/CSE8803-DLT/C4_data_100samples/labels.pt')

    # # Remove the last row from each tensor using list comprehension as the last token does NOT have a next word prediction label
    # feats_actual = [tensor[:,:-1,:] for tensor in feats_actual]
    # dola_actual = [tensor[:-1,:] for tensor in dola_actual]

    # # Reshape the dataset components appropriately for epinet training
    # feats_actual = torch.cat(feats_actual, dim=1)
    # feats_actual = feats_actual.reshape(-1, input_dim)      # (num_samples, input_dim)
    # feats_actual = feats_actual.cpu().detach().numpy()

    # dola_actual = torch.cat(dola_actual, dim=0)             # (num_samples, 32000)
    # dola_actual = dola_actual.cpu().detach().numpy()
    # if not config['use_dola_raw_logits']:
    #     dola_actual = jax.nn.softmax(dola_actual)               # Convert the DoLa logits into softmax distributions

    # labels_actual = torch.cat(labels_actual, dim=1)         # (1, num_samples)
    # labels_actual = labels_actual.squeeze(0).cpu().detach().numpy()

    # feats_actual = feats_actual[:num_batch * batch_size,:]
    # dola_actual = dola_actual[:num_batch * batch_size,:]
    # labels_actual = labels_actual[:num_batch * batch_size]

    # print("\n Dummy dataset shapes: ")
    # print("x.shape: ", x.shape)
    # print("y.shape: ", y.shape)
    # print("dola dist shape: ", dola_distribution.shape)

    # print("\n Actual dataset shapes: ")
    # print("feats_actual.shape: ", feats_actual.shape)
    # print("labels_actual.shape: ", labels_actual.shape)
    # print("dola_actual shape: ", dola_actual.shape)

    return utils.make_batch_iterator(data=datasets.ArrayBatch(x=x, 
                                                         y=y, 
                                                         extra={"dola_distribution": dola_distribution}), 
                                     batch_size=batch_size)

class Epinet:
    def __init__(self, 
                 input_size, 
                 output_size, 
                 index_dim,
                 epinet_hiddens,
                 pretrained_params_file):
        dummy_vocab_head = jax.random.uniform(jax.random.PRNGKey(42), shape=(input_size, output_size))
        vocab_head = functools.partial(projection_layer, 
                            feature_size=input_size,
                            logit_size=output_size,
                            vocab_head_weight=dummy_vocab_head)
        self.epinet = MLPEpinetWithTrainableAndPrior(
                                                projection_layer=vocab_head,
                                                index_dim=index_dim,
                                                num_classes=output_size,
                                                epinet_hiddens=epinet_hiddens)
        
        with open(pretrained_params_file, 'rb') as f:
            self.params = dill.load(f)

    
    def apply(self, inputs, rng):
        index = self.epinet.indexer(next(rng))
        return self.epinet.apply(self.params, {}, inputs, index)



