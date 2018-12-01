export function repeat<T>(a: T, count: number): T[] {
  return new Array(count).fill(0).map(() => a);
}
