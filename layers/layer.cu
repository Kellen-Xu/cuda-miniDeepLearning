#include "layer.h"

void Layer::setInputLayer(Layer *prev_layer)
{
    this->prev_output = prev_layer->output;
    this->prev_d_output = prev_layer->d_output;

    this->isPrevInput = (prev_layer->type == INPUT_LAYER_TYPE);
}