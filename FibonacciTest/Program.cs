using System;
using System.Diagnostics;

namespace FibonacciTest
{
    static class Program
    {
        static void Main(string[] args)
        {
            Fibonacci_Methods();
            Console.ReadLine();
        }

        delegate long Fib(int N, long[] D = null);
        static Fib[] Fibonaci_Methods = { DOWN, UP, Divide_Conquer };
        static int[] D, D1;
        static int sum;

        //divide-and-conquer
        static long Divide_Conquer(int N, long[] D = null)
        {
            switch (N)
            {
                case 0:
                case 1:
                    return 1;
                default:
                    return Divide_Conquer(N - 2) + Divide_Conquer(N - 1);
            }
        }

        //bottom-down method
        static long DOWN(int N, long[] D = null)
        {
            if (D[N] != 0) return D[N];
            switch (N)
            {
                case 0:
                    D[N] = 0;
                    break;
                case 1:
                    D[N] = 1;
                    break;
                default:
                    D[N] = DOWN(N - 1, D) + DOWN(N - 2, D);
                    break;
            }
            return D[N];
        }

        //bottom-up method
        static long UP(int N, long[] D)
        {
            for (var i = 2; i < N; i++)
                D[i] = D[i - 1] + D[i - 2];
            return D[N];
        }
        static double Bine(int N, long[] D = null)
        {
            var phi1 = Math.Pow(((1 + Math.Sqrt(5)) * 0.5), N - 1);
            var phi2 = Math.Pow(((1 - Math.Sqrt(5)) * 0.5), N - 1);
            return (phi1 - phi2) / Math.Sqrt(5);
        }
        static void Fibonacci_Methods()
        {
            Console.Write("Set Size : ");
            var size = int.Parse(Console.ReadLine()) + 1;
            var D = new long[size];
            D[0] = 0;
            D[1] = 1;
            var Time = new Stopwatch();
            //------------------- Бине --------------------
            Time.Reset(); Time.Start();
            var g = Bine(size);
            Console.Write($"{"Bine",14} F ={g,7:F0}");
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write($"| Elapsed ={ Time.ElapsedTicks,7} Ticks\n");
            Console.ResetColor();
            Time.Stop();
            //---------------------------------------------
            foreach (var m in Fibonaci_Methods)
            {
                Time.Reset(); Time.Start();
                long f;
                if (m.Method.Name == "DOWN" || m.Method.Name == "Divide_Conquer") f = m(--size, D);
                f = m(size, D);
                Time.Stop();
                Console.Write($"{m.Method.Name,14} F ={f,7}");
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"| Elapsed ={ Time.ElapsedTicks,7} Ticks\n");
                Console.ResetColor();
            }
        }
        //---------------------------------------------
    }
}
