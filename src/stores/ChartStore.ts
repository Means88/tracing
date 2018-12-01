import { action, observable } from "mobx";

class ChartStore {
  @observable public minValue: number = 0;
  @observable public maxValue: number = 0;

  @action public setMinValue(x: number) {
    this.minValue = Math.max(x, 0);
  }

  @action public setMaxValue(x: number) {
    this.maxValue = x;
  }
}

export default new ChartStore();
